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


""""
General components of a model

WeightPredictorNetwork(optional) -> FeatureExtractor -> Decoder (optional) -> DNN/DKL
"""
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


class WeightPredictorNetwork(nn.Module):
	"""
	Linear -> Tanh -> Linear -> Tanh
	"""
	def __init__(self, args, embedding_matrix):
		"""
		A tiny network outputs that outputs a matrix W.

		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		print(f"Initializing WeightPredictorNetwork with embedding_matrix of size {embedding_matrix.size()}")
		self.args = args

		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix


		##### Weight predictor network (wpn)
		layers = []
		prev_dimension = args.wpn_embedding_size
		for i, dim in enumerate(args.diet_network_dims):
			if i == len(args.diet_network_dims)-1: # last layer
				layer = nn.Linear(prev_dimension, dim)
				nn.init.uniform_(layer.weight, -0.01, 0.01) # same initialization as in the DietNetwork original paper
				layers.append(layer)
				layers.append(nn.Tanh())
			else:
				if args.nonlinearity_weight_predictor=='tanh':
					layer = nn.Linear(prev_dimension, dim)
					nn.init.uniform_(layer.weight, -0.01, 0.01) # same initialization as in from the DietNetwork original paper
					layers.append(layer)
					layers.append(nn.Tanh())					# DietNetwork paper uses tanh all over
				elif args.nonlinearity_weight_predictor=='leakyrelu':
					layer = nn.Linear(prev_dimension, dim)
					nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
					layers.append(layer)
					layers.append(nn.LeakyReLU())

				if args.batchnorm:
					layers.append(nn.BatchNorm1d(dim))
				layers.append(nn.Dropout(args.dropout_rate))
				
			prev_dimension = dim

		self.wpn = nn.Sequential(*layers)


		#### Residual embeddings
		self.nn_residual_embedding = None
		if args.residual_embedding=='resnet':
			print(f"Using ---{args.residual_embedding}--- residual embeddings")

			# use the updated ResNet order, including dropout because we use Linear layers instead of Convolutional
			linear_1 = nn.Linear(args.wpn_embedding_size, args.wpn_embedding_size)
			linear_2 = nn.Linear(args.wpn_embedding_size, args.wpn_embedding_size)

			# initialize the layers
			nn.init.kaiming_normal_(linear_1.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
			nn.init.kaiming_normal_(linear_2.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')

			self.nn_residual_embedding = nn.Sequential(
				nn.BatchNorm1d(args.wpn_embedding_size),
				nn.Dropout(args.dropout_rate),
				nn.LeakyReLU(),
				linear_1,

				nn.BatchNorm1d(args.wpn_embedding_size),
				nn.Dropout(args.dropout_rate),
				nn.LeakyReLU(),
				linear_2,
			)


	def forward(self):
		# use the wpn to predict the weight matrix
		embeddings = self.embedding_matrix
		if self.nn_residual_embedding:
			embeddings = embeddings + self.nn_residual_embedding(embeddings)

		W = self.wpn(embeddings) # W has size (D x K)
		
		if self.args.softmax_diet_network:
			W = F.softmax(W, dim=1) # FsNet applied softmax over the (feature - all-K-neurons)
		
		return W.T # size K x D


class ConcreteLayer(nn.Module):
	"""
	Implementation of a concrete layer from paper "Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"
	"""

	def __init__(self, args, input_dim, output_dim, is_diet_layer=False, wpn_embedding_matrix=None):
		"""
		- input_dim (int): dimension of the input
		- output_dim (int): number of neurons in the layer
		"""
		super().__init__()
		self.args = args
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.temp_start = 10
		self.temp_end = 0.01
		# the iteration is used in annealing the temperature
		# 	it's increased with every call to sample during training
		self.current_iteration = 0 
		self.anneal_iterations = args.concrete_anneal_iterations # maximum number of iterations for the temperature optimization

		self.is_diet_layer = is_diet_layer
		if is_diet_layer:
			# if diet layer, then initialize a weight predictor matrix
			assert wpn_embedding_matrix is not None
			self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
		else:
			# alphas (output_dim x input_dim) - learnable parameters for each neuron
			# alphas[i] = parameters of neuron i
			self.alphas = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=True)
			torch.nn.init.xavier_normal_(self.alphas, gain=1) # Glorot normalization, following the original CAE implementation
		
	def get_temperature(self):
		# compute temperature		
		if self.current_iteration >= self.anneal_iterations:
			return self.temp_end
		else:
			return self.temp_start * (self.temp_end / self.temp_start) ** (self.current_iteration / self.anneal_iterations)

	def sample(self):
		"""
		Sample from the concrete distribution.
		"""
		# Increase the iteration counter during training
		if self.training:
			self.current_iteration += 1

		temperature = self.get_temperature()

		alphas = self.wpn() if self.is_diet_layer else self.alphas # alphas is a K x D matrix

		# sample from the concrete distribution
		if self.training:
			samples = F.gumbel_softmax(alphas, tau=temperature, hard=False) # size K x D
			assert samples.shape == (self.output_dim, self.input_dim)
		else: 			# sample using argmax
			index_max_alphas = torch.argmax(alphas, dim=1) # size K
			samples = torch.zeros(self.output_dim, self.input_dim).cuda()
			samples[torch.arange(self.output_dim), index_max_alphas] = 1.

		return samples

	def forward(self, x):
		"""
		- x (batch_size x input_dim)
		"""
		mask = self.sample()   	# size (number_neurons x input_dim)
		x = torch.matmul(x, mask.T) 		# size (batch_size, number_neurons)
		return x, None # return additional None for compatibility


def create_linear_layers(args, layer_sizes, layers_for_hidden_representation):
	"""
	Args
	- layer_sizes: list of the sizes of the sizes of the linear layers
	- layers_for_hidden_representation: number of layers of the first part of the encoder (used to output the input for the decoder)

	Returns
	Two lists of Pytorch Modules (e.g., Linear, BatchNorm1d, Dropout)
	- encoder_first_part
	- encoder_second_part
	"""
	encoder_first_part = []
	encoder_second_part = []
	for i, (dim_prev, dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
		if i < layers_for_hidden_representation:					# first part of the encoder
			encoder_first_part.append(nn.Linear(dim_prev, dim))
			encoder_first_part.append(nn.LeakyReLU())
			if args.batchnorm:
				encoder_first_part.append(nn.BatchNorm1d(dim))
			encoder_first_part.append(nn.Dropout(args.dropout_rate))
		else:														# second part of the encoder
			encoder_second_part.append(nn.Linear(dim_prev, dim))
			encoder_second_part.append(nn.LeakyReLU())
			if args.batchnorm:
				encoder_second_part.append(nn.BatchNorm1d(dim))
			encoder_second_part.append(nn.Dropout(args.dropout_rate))
		
	return encoder_first_part, encoder_second_part


class FirstLinearLayer(nn.Module):
	"""
	First linear layer (with activation, batchnorm and dropout), with the ability to include:
	- diet layer (i.e., there's a weight predictor network which predicts the weight matrix)
	- sparsity network (i.e., there's a sparsity network which outputs sparsity weights)
	"""

	def __init__(self, args, is_diet_layer, sparsity_type, wpn_embedding_matrix, spn_embedding_matrix):
		"""
		If is_diet_layer==None and sparsity_type==None, this layers acts as a standard linear layer
		"""
		super().__init__()

		self.args = args
		self.is_diet_layer = is_diet_layer
		self.sparsity_type = sparsity_type

		# DIET LAYER
		if is_diet_layer:
			# if diet layer, then initialize a weight predictor network
			self.wpn = WeightPredictorNetwork(args, wpn_embedding_matrix)
		else:
			# standard linear layer
			self.weights_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0], args.num_features))
			nn.init.kaiming_normal_(self.weights_first_layer, a=0.01, mode='fan_out', nonlinearity='leaky_relu')

		# auxiliary layer after the matrix multiplication
		self.bias_first_layer = nn.Parameter(torch.zeros(args.feature_extractor_dims[0]))
		self.layers_after_matrix_multiplication = nn.Sequential(*[
			nn.LeakyReLU(),
			nn.BatchNorm1d(args.feature_extractor_dims[0]),
			nn.Dropout(args.dropout_rate)
		])

		# SPARSITY REGULARIZATION for the first layer
		if sparsity_type=='global':
			if args.sparsity_method=='sparsity_network':
				print("Creating Sparsity network")
				self.sparsity_model = SparsityNetwork(args, spn_embedding_matrix)
			elif args.sparsity_method=='learnable_vector':
				print("Creating learnable network")
				self.sparsity_model = LearnableSparsityVector(args)
			else:
				raise Exception("Sparsity method not valid")
		elif args.sparsity_type=='local':
			if args.sparsity_method=='sparsity_network':
				self.sparsity_model = SparsityNetwork(args, spn_embedding_matrix)
			else:
				raise Exception("Sparsity method not valid")
		else:
			self.sparsity_model = None

	def forward(self, x):
		"""
		Input:
			x: (batch_size x num_features)
		"""
		# first layer
		W = self.wpn() if self.is_diet_layer else self.weights_first_layer # W has size (K x D)
		
		if self.args.sparsity_type==None:
			all_sparsity_weights = None

			hidden_rep = F.linear(x, W, self.bias_first_layer)
		
		elif self.args.sparsity_type=='global':
			all_sparsity_weights = self.sparsity_model(None) 	# Tensor (D, )
			assert all_sparsity_weights.shape[0]==self.args.num_features and len(all_sparsity_weights.shape)==1
			W = torch.matmul(W, torch.diag(all_sparsity_weights))

			hidden_rep = F.linear(x, W, self.bias_first_layer)

		elif self.args.sparsity_type=='local':
			# create a different weight matrix for each sample
			hidden_rep = torch.zeros(x.shape[0], self.args.feature_extractor_dims[0], device=x.device)
			
			all_sparsity_weights = self.sparsity_model(x) # size (B, D)
			assert all_sparsity_weights.shape[0]==x.shape[0] and all_sparsity_weights.shape[1]==self.args.num_features
			
			#### Slow way to compute the hidden representation
			for i, x_i in enumerate(x): # x_i has size (D)
				sparsity_weights = all_sparsity_weights[i]
				assert sparsity_weights.shape[0]==self.args.num_features and len(sparsity_weights.shape)==1

				W = W * sparsity_weights

				hidden_rep_x_i = F.linear(torch.unsqueeze(x_i, dim=0), W, self.bias_first_layer) # (1 x out_dimension)
				hidden_rep_x_i = torch.squeeze(hidden_rep_x_i, dim=0) # (out_dimension)

				assert len(hidden_rep_x_i.shape)==1 and hidden_rep_x_i.shape[0]==self.args.feature_extractor_dims[0]

				hidden_rep[i] = hidden_rep_x_i
			
	
		return self.layers_after_matrix_multiplication(hidden_rep), all_sparsity_weights


class Decoder(nn.Module):
	def __init__(self, args, wpn):
		super().__init__()
		assert wpn!=None, "The decoder is used only with a WPN (because it's only used within the DietNetwork)"

		self.wpn = wpn
		self.bias = nn.Parameter(torch.zeros(args.num_features,))

	def forward(self, hidden_rep):
		W = self.wpn().T # W has size D x K

		return F.linear(hidden_rep, W, self.bias)


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
		
		# log temperature of the concrete distribution
		if isinstance(self.first_layer, ConcreteLayer):
			self.log("train/concrete_temperature", self.first_layer.get_temperature())

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


class DNN(TrainingLightningModule):
	"""
	Flexible MLP-based architecture which can implement an MLP, WPS, FsNet


	DietDNN architecture
	Linear -> LeakyRelu -> BatchNorm -> Dropout -> Linear -> LeakyRelu-> BatchNorm -> Dropout -> Linear -> y_hat
																							|
																							|
																						  hidden
																							|
																							v
																						  Linear
																							|
																							V
																						  x_hat
	"""
	def __init__(self, args, first_layer, decoder):
		"""
		DNN with one last layer added (with `num_classes` logits) on top of the feature_extractor.
		It's fair to add one more layer, because the DKL has a mixing layer added after the GP.

		:param nn.Module feature_extractor: the feature_extractor (except the last layer to the softmax logits)
		:param nn.Module decoder: decoder (for reconstruction loss)
				If None, then don't have a reconstruction loss
		"""
		super().__init__(args)

		if decoder:
			print(f'Creating {args.model} with decoder...')
		else:
			print(f'Creating {args.model} without decoder...')

		self.args = args
		self.log_test_key = None
		self.learning_rate = args.lr
		
		self.first_layer = first_layer
		encoder_first_layers, encoder_second_layers = create_linear_layers(
			args, args.feature_extractor_dims, args.layers_for_hidden_representation-1) # the -1 in (args.layers_for_hidden_representation - 1) is because we don't consider the first layer

		self.encoder_first_layers = nn.Sequential(*encoder_first_layers)
		self.encoder_second_layers = nn.Sequential(*encoder_second_layers)

		self.classification_layer = nn.Linear(args.feature_extractor_dims[-1], args.num_classes)
		self.decoder = decoder

	def forward(self, x):
		x, sparsity_weights = self.first_layer(x)			   # pass through first layer

		x = self.encoder_first_layers(x)					   # pass throught the first part of the following layers
		x_hat = self.decoder(x) if self.decoder else None      # reconstruction

		x = self.encoder_second_layers(x)
		y_hat = self.classification_layer(x)           		   # classification, returns logits
		
		return y_hat, x_hat, sparsity_weights