# create the GP layer called after the neural network
# using **one** GP per feature (as in the SV-DKL paper)
### the outputs of these GPs will be mixed in the softmax likelihood
from json import encoder
from math import e, gamma
from modulefinder import STORE_OPS
from grpc import xds_channel_credentials
from importlib_metadata import version
from pyro import param, sample
import scipy
from sklearn.utils import axis0_safe_slice
from torch.nn.functional import embedding
import wandb
from _shared_imports import *
from torch import nn

from lookahead_optimizer import Lookahead
from sparsity import LearnableSparsityVector, SparsityNetwork


def get_labels_lists(outputs):
	all_y_true, all_y_pred = [], []
	for output in outputs:
		all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
		all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

	return all_y_true, all_y_pred


def compute_all_metrics(args, y_true, y_pred):
	metrics = {}
	metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
	metrics['F1_weighted'] = f1_score(y_true, y_pred, average='weighted')
	metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
	metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
	if args.num_classes==2:
		metrics['AUROC_weighted'] = roc_auc_score(y_true, y_pred, average='weighted')
	
	return metrics


def detach_tensors(tensors):
	"""
	Detach losses 
	"""
	if type(tensors)==list:
		detached_tensors = list()
		for tensor in tensors:
			detach_tensors.append(tensor.detach())
	elif type(tensors)==dict:
		detached_tensors = dict()
		for key, tensor in tensors.items():
			detached_tensors[key] = tensor.detach()
	else:
		raise Exception("tensors must be a list or a dict")
	
	return detached_tensors

def reshape_batch(batch):
	"""
	When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
	
	This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
	"""
	x, y = batch
	x = x.reshape(-1, x.shape[-1])
	y = y.reshape(-1)

	return x, y



def create_model(args, data_module=None):
	"""
	Function to create the model. Firstly creates the components (e.g., FeatureExtractor, Decoder) and then assambles them.

	Returns a model instance.
	"""
	pl.seed_everything(args.seed_model_init, workers=True)
	
	### create embedding matrices
	wpn_embedding_matrix = data_module.get_embedding_matrix(args.wpn_embedding_type, args.wpn_embedding_size)
	if args.wpn_embedding_type==args.sparsity_gene_embedding_type and args.wpn_embedding_size==args.sparsity_gene_embedding_size:
		spn_embedding_matrix = wpn_embedding_matrix
	else:
		spn_embedding_matrix = data_module.get_embedding_matrix(args.sparsity_gene_embedding_type, args.sparsity_gene_embedding_size)

	### create decoder
	if args.gamma > 0:
		wpn_decoder = WeightPredictorNetwork(args, wpn_embedding_matrix)
		decoder = Decoder(args, wpn_decoder)
	else:
		decoder = None

	### create models
	if args.model=='fsnet':
		concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0], is_diet_layer=True, wpn_embedding_matrix=wpn_embedding_matrix)

		model = DNN(args, concrete_layer, decoder)

	elif args.model=='cae': # Supervised Autoencoder
		concrete_layer = ConcreteLayer(args, args.num_features, args.feature_extractor_dims[0])

		model = DNN(args, concrete_layer, None)

	elif args.model in ['dnn', 'dietdnn']:
		if args.model=='dnn':
			is_diet_layer = False
		elif args.model=='dietdnn':
			is_diet_layer = True
		
		first_layer = FirstLinearLayer(args, is_diet_layer=is_diet_layer, sparsity_type=args.sparsity_type,
						wpn_embedding_matrix=wpn_embedding_matrix, spn_embedding_matrix=spn_embedding_matrix)

		model = DNN(args, first_layer, decoder)
	else:
		raise Exception(f"The model ${args.model}$ is not supported")

	return model



""""
Metrics
- all 
	- balanced_accuracy
	- F1 - weighted
	- precision - weighted
	- recall - weighted
	- accuracy per class
- binary
	- AUROC (binary)
- loss
	- total
	- reconstruction
	- DNN
		- cross-entropy
	- DKL
		- data fit
		- complexity penalty
"""

class TrainingLightningModule(pl.LightningModule):
	"""
	General class to be inherited by all implemented models (e.g., MLP, CAE, FsNet etc.)

	It implements general training and evaluation functions (e.g., computing losses, logging, training etc.)
	"""
	def __init__(self, args):
		super().__init__()
		self.args = args

	def compute_loss(self, y_true, y_hat, x, x_hat, sparsity_weights):
		losses = {}
		losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))
		losses['reconstruction'] = self.args.gamma * F.mse_loss(x_hat, x, reduction='mean') if self.decoder else torch.zeros(1, device=self.device)

		### sparsity loss
		if sparsity_weights is None:
			losses['sparsity'] = torch.tensor(0., device=self.device)
		else:
			if self.args.sparsity_regularizer=='L1':
				losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * torch.norm(sparsity_weights, 1)
			elif self.args.sparsity_regularizer=='hoyer':
				hoyer_reg = torch.norm(sparsity_weights, 1) / torch.norm(sparsity_weights, 2)
				losses['sparsity'] = self.args.sparsity_regularizer_hyperparam * hoyer_reg
			else:
				raise Exception("Sparsity regularizer not valid")

		losses['total'] = losses['cross_entropy'] + losses['reconstruction'] + losses['sparsity']
		
		return losses

	def log_losses(self, losses, key, dataloader_name=""):
		self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
		self.log(f"{key}/reconstruction_loss{dataloader_name}", losses['reconstruction'].item())
		self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item())
		self.log(f"{key}/sparsity_loss{dataloader_name}", losses['sparsity'].item())

	def log_epoch_metrics(self, outputs, key, dataloader_name=""):
		y_true, y_pred = get_labels_lists(outputs)
		self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred))
		self.log(f'{key}/F1_weighted{dataloader_name}', f1_score(y_true, y_pred, average='weighted'))
		self.log(f'{key}/precision_weighted{dataloader_name}', precision_score(y_true, y_pred, average='weighted'))
		self.log(f'{key}/recall_weighted{dataloader_name}', recall_score(y_true, y_pred, average='weighted'))
		if self.args.num_classes==2:
			self.log(f'{key}/AUROC_weighted{dataloader_name}', roc_auc_score(y_true, y_pred, average='weighted'))

	def training_step(self, batch, batch_idx):
		x, y_true = batch
	
		y_hat, x_hat, sparsity_weights = self.forward(x)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		self.log_losses(losses, key='train')
		# self.log("train/lr", self.learning_rate)
		
		return {
			'loss': losses['total'],
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}

	def training_epoch_end(self, outputs):
		self.log_epoch_metrics(outputs, 'train')

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		"""
		- dataloader_idx (int) tells which dataloader is the `batch` coming from
		"""
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x)

		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		return {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}

	def validation_epoch_end(self, outputs_all_dataloaders):
		"""
		- outputs: when no_dataloaders==1 --> A list of dictionaries corresponding to a validation step.
				   when no_dataloaders>1  --> List with length equal to the number of validation dataloaders. Each element is a list with the dictionaries corresponding to a validation step.
		"""
		### Log losses and metrics
		# `outputs_all_dataloaders` is expected to a list of dataloaders.
		# However, when there's only one dataloader, outputs_all_dataloaders is NOT a list.
		# Thus, we transform it in a list to preserve compatibility
		if len(self.args.val_dataloaders_name)==1:
			outputs_all_dataloaders = [outputs_all_dataloaders]

		for dataloader_id, outputs in enumerate(outputs_all_dataloaders):
			losses = {
				'total': np.mean([output['losses']['total'].item() for output in outputs]),
				'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
				'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
				'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
			}
			if dataloader_id==0: # original validation dataset
				dataloader_name=""
			else:
				dataloader_name=f"__{self.args.val_dataloaders_name[dataloader_id]}"

			self.log_losses(losses, key='valid', dataloader_name=dataloader_name)
			self.log_epoch_metrics(outputs, key='valid', dataloader_name=dataloader_name)


	def test_step(self, batch, batch_idx, dataloader_idx=0):
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x)
		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		return {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1),
			'y_hat': y_hat.detach().cpu().numpy()
		}

	def test_epoch_end(self, outputs):
		### Save losses
		losses = {
			'total': np.mean([output['losses']['total'].item() for output in outputs]),
			'reconstruction': np.mean([output['losses']['reconstruction'].item() for output in outputs]),
			'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
			'sparsity': np.mean([output['losses']['sparsity'].item() for output in outputs])
		}
		self.log_losses(losses, key=self.log_test_key)
		self.log_epoch_metrics(outputs, self.log_test_key)

		#### Save prediction probabilities
		y_hat_list = [output['y_hat'] for output in outputs]
		y_hat_all = np.concatenate(y_hat_list, axis=0)
		y_hat_all = scipy.special.softmax(y_hat_all, axis=1)

		y_hat_all = wandb.Table(dataframe=pd.DataFrame(y_hat_all))
		wandb.log({f'{self.log_test_key}_y_hat': y_hat_all})


		### Save global feature importances
		if self.args.sparsity_type == 'global':
			feature_importance = self.feature_extractor.sparsity_model.forward(None).cpu().detach().numpy()
			
			global_feature_importance = wandb.Table(dataframe=pd.DataFrame(feature_importance))
			wandb.log({f'{self.log_test_key}_global_feature_importance': global_feature_importance})


	def configure_optimizers(self):
		params = self.parameters()

		if self.args.optimizer=='adam':
			optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
		if self.args.optimizer=='adamw':
			optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])
		
		if self.args.lookahead_optimizer:
			optimizer = Lookahead(optimizer, la_steps=5, la_alpha=0.5)

		if self.args.lr_scheduler == None:
			return optimizer
		else:
			if self.args.lr_scheduler == 'plateau':
				lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
			elif self.args.lr_scheduler == 'cosine_warm_restart':
				# Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
				# 	splits the scheduler for 6 periods. We split into 6 periods as well.
				lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
					T_0 = self.args.cosine_warm_restart_t_0,
					eta_min = self.args.cosine_warm_restart_eta_min,
					verbose=True)
			elif self.args.lr_scheduler == 'linear':
				lr_scheduler = torch.optim.lr_scheduler.LinearLR(
					optimizer, 
					start_factor = self.args.lr,
					end_factor = 3e-5,
					total_iters = self.args.max_steps / self.args.val_check_interval)
			elif self.args.lr_scheduler == 'lambda':
				def scheduler(epoch):
					if epoch < 500:
						return 0.995 ** epoch
					else:
						return 0.1

				lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
					optimizer,
					scheduler)
			else:
				raise Exception()

			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': lr_scheduler,
					'monitor': 'valid/cross_entropy_loss',
					'interval': 'step',
					'frequency': self.args.val_check_interval,
					'name': 'lr_scheduler'
				}
			}
