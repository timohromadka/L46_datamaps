import numpy as np
import collections
from dataclasses import dataclass
from pickletools import optimize
from statistics import mode
import pytorch_lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from lassonet import LassoNetClassifier

import wandb
import json
import pprint
import warnings
import sklearn
import logging

from args import parser
from dataset import *
from models import *
from _config import DATA_DIR



def main():
    
	warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
	warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.LightningDeprecationWarning)

	print("Starting...")

	logging.basicConfig(
		filename='/home/am2770/Github/cancer-low-data/logs_exceptions.txt',
		filemode='a',
		format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
		datefmt='%H:%M:%S',
		level=logging.DEBUG
	)

 	args = parser.parse_args()



	#### Assert sparsity parameters
	if args.sparsity_type:
		# if one of the sparsity parameters is set, then all of them must be set
		assert args.sparsity_gene_embedding_type
		assert args.sparsity_type
		assert args.sparsity_method
		assert args.sparsity_regularizer
		# assert args.sparsity_regularizer_hyperparam

	# add best performing configuration
	if args.use_best_hyperparams:
		# if the model uses gene embeddings of any type, then use dataset specific embedding sizes.
		if args.model in ['fsnet', 'dietdnn']:
			if args.dataset=='cll':
				args.wpn_embedding_size = 70
			elif args.dataset=='lung':
				args.wpn_embedding_size = 20
			else:
				args.wpn_embedding_size = 50

		if args.sparsity_type in ['global', 'local']:
			if args.dataset=='cll':
				args.sparsity_gene_embedding_size = 70
			elif args.dataset=='lung':
				args.sparsity_gene_embedding_size = 20
			else:
				args.sparsity_gene_embedding_size = 50

		elif args.model=='rf':
			params = {
				'cll': (3, 3),
				'lung': (3, 2),
				'metabric-dr': (7, 2),
				'metabric-pam50': (7, 2),
				'prostate': (5, 2),
				'smk': (5, 2),
				'tcga-2ysurvival': (3, 3),
				'tcga-tumor-grade': (3, 3),
				'toxicity': (5, 3)
			}

			args.rf_max_depth, args.rf_min_samples_leaf = params[args.dataset]

		elif args.model=='lasso':
			params = {
				'cll': 10,
				'lung': 100,
				'metabric-dr': 100,
				'metabric-pam50': 10,
				'prostate': 100,
				'smk': 1000,
				'tcga-2ysurvival': 10,
				'tcga-tumor-grade': 100,
				'toxicity': 100
			}

			args.lasso_C = params[args.dataset]
		
		elif args.model=='tabnet':
			params = {
				'cll': (0.03, 0.001),
				'lung': (0.02, 0.1),
				'metabric-dr': (0.03, 0.1),
				'metabric-pam50': (0.02, 0.001),
				'prostate': (0.02, 0.01),
				'smk': (0.03, 0.001),
				'tcga-2ysurvival': (0.02, 0.01),
				'tcga-tumor-grade': (0.02, 0.01),
				'toxicity': (0.03, 0.1)
			}

			args.lr, args.tabnet_lambda_sparse = params[args.dataset]

		elif args.model=='lgb':
			params = {
				'cll': (0.1, 2),
				'lung': (0.1, 1),
				'metabric-dr': (0.1, 1),
				'metabric-pam50': (0.01, 2),
				'prostate': (0.1, 2),
				'smk': (0.1, 2),
				'tcga-2ysurvival': (0.1, 1),
				'tcga-tumor-grade': (0.1, 1),
				'toxicity': (0.1, 2)
			}

			args.lgb_learning_rate, args.lgb_max_depth = params[args.dataset]
		

	run_experiment(args)


def run_experiment(args):
	args.suffix_wand_run_name = f"repeat-{args.repeat_id}__test-{args.test_split}"

	#### Load dataset
	print(f"\nInside training function")
	print(f"\nLoading data {args.dataset}...")
	data_module = create_data_module(args)
	
	print(f"Train/Valid/Test splits of sizes {args.train_size}, {args.valid_size}, {args.test_size}")
	print(f"Num of features: {args.num_features}")

	#### Intialize logging
	wandb_logger = create_wandb_logger(args)

	wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"


	#### Scikit-learn training
	if args.model in ['lasso', 'rf', 'lgb', 'tabnet', 'lassonet']:
		# scikit-learn expects class_weights to be a dictionary
		class_weights = {}
		for i, val in enumerate(args.class_weights):
			class_weights[i] = val

		class_weights_list = [class_weights[i] for i in range(len(class_weights))]
		

		if args.model == 'lasso':
			model = LogisticRegression(penalty='l1', C=args.lasso_C, 
						class_weight=class_weights, max_iter=10000,
						random_state=42, solver='saga', verbose=True)
			model.fit(data_module.X_train, data_module.y_train)

		elif args.model == 'rf':
			model = RandomForestClassifier(n_estimators=args.rf_n_estimators, 
						min_samples_leaf=args.rf_min_samples_leaf, max_depth=args.rf_max_depth,
						class_weight=class_weights, max_features='sqrt',
						random_state=42, verbose=True)
			model.fit(data_module.X_train, data_module.y_train)

		elif args.model == 'lgb':
			params = {
				'max_depth': args.lgb_max_depth,
				'learning_rate': args.lgb_learning_rate,
				'min_data_in_leaf': args.lgb_min_data_in_leaf,

				'class_weight': class_weights,
				'n_estimators': 200,
				'objective': 'cross_entropy',
				'num_iterations': 10000,
				'device': 'gpu',
				'feature_fraction': '0.3'
			}

			model = lgb.LGBMClassifier(**params)
			model.fit(data_module.X_train, data_module.y_train,
		  		eval_set=[(data_module.X_valid, data_module.y_valid)],
	 	  		callbacks=[lgb.early_stopping(stopping_rounds=100)])

		elif args.model == 'tabnet':
			model = TabNetClassifier(
				n_d=8, n_a=8, # The TabNet implementation says "Bigger values gives more capacity to the model with the risk of overfitting"
				n_steps=3, gamma=1.5, n_independent=2, n_shared=2, # default values
				momentum=0.3, clip_value=2.,
				lambda_sparse=args.tabnet_lambda_sparse,
				optimizer_fn=torch.optim.Adam,
				optimizer_params=dict(lr=args.lr), # the paper sugests 2e-2
				scheduler_fn=torch.optim.lr_scheduler.StepLR,
				scheduler_params = {"gamma": 0.95, "step_size": 20},
				seed=args.seed_training
			)

			class WeightedCrossEntropy(Metric):
				def __init__(self):
					self._name = "cross_entropy"
					self._maximize = False

				def __call__(self, y_true, y_score):
					aux = F.cross_entropy(
						input=torch.tensor(y_score, device='cuda'),
						target=torch.tensor(y_true, device='cuda'),
						weight=torch.tensor(args.class_weights, device='cuda')
					).detach().cpu().numpy()

					return float(aux)

			virtual_batch_size = 5
			if args.dataset == 'lung': 
				virtual_batch_size = 6 # lung has training of size 141. With a virtual_batch_size of 5, the last batch is of size 1 and we get an error because of BatchNorm

			batch_size = args.train_size
			model.fit(data_module.X_train, data_module.y_train,
		  			  eval_set=[(data_module.X_valid, data_module.y_valid)],
					  eval_metric=[WeightedCrossEntropy], 
					  loss_fn=torch.nn.CrossEntropyLoss(torch.tensor(args.class_weights, device='cuda')),
					  batch_size=batch_size,
					  virtual_batch_size=virtual_batch_size,
					  max_epochs=5000, patience=100)
		
		elif args.model == 'lassonet':
			model = LassoNetClassifier(
				lambda_start=args.lassonet_lambda_start,
				gamma=args.lassonet_gamma,
				gamma_skip=args.lassonet_gamma,
				M = args.lassonet_M,
				n_iters = args.lassonet_epochs,
				
				optim = partial(torch.optim.AdamW, lr=1e-4, betas=[0.9, 0.98]),
				hidden_dims=(100, 100, 10),
				class_weight = class_weights_list, # use weighted loss
				dropout=0.2,
				batch_size=8,
				backtrack=True # if True, ensure the objective is decreasing
				# random_state = 42, # seed for validation set, 
									 # no need to use because we provide validation set
			)

			model.path(data_module.X_train, data_module.y_train,
				X_val = data_module.X_valid, y_val = data_module.y_valid)


		#### Log metrics
		y_pred_train = model.predict(data_module.X_train)
		y_pred_valid = model.predict(data_module.X_valid)
		y_pred_test = model.predict(data_module.X_test)

		train_metrics = compute_all_metrics(args, data_module.y_train, y_pred_train)
		valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
		test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)

		for metrics, dataset_name in zip(
			[train_metrics, valid_metrics, test_metrics],
			["bestmodel_train", "bestmodel_valid", "bestmodel_test"]):
			for metric_name, metric_value in metrics.items():
				wandb.run.summary[f"{dataset_name}/{metric_name}"] = metric_value

	#### Pytorch lightning training
	else:

		#### Set embedding size if it wasn't provided
		if args.wpn_embedding_size==-1:
			args.wpn_embedding_size = args.train_size
		if args.sparsity_gene_embedding_size==-1:
			args.sparsity_gene_embedding_size = args.train_size

		args.num_tasks = args.feature_extractor_dims[-1] 			# number of output units of the feature extractor. Used for convenience when defining the GP

		
		if args.max_steps!=-1:
			# compute the upper rounded number of epochs to training (used for lr scheduler in DKL)
			steps_per_epoch = np.floor(args.train_size / args.batch_size)
			args.max_epochs = int(np.ceil(args.max_steps / steps_per_epoch))
			print(f"Training for max_epochs = {args.max_epochs}")


		#### Create model
		model = create_model(args, data_module)

		trainer, checkpoint_callback = train_model(args, model, data_module, wandb_logger)

		if args.train_on_full_data:
			checkpoint_path = checkpoint_callback.last_model_path
		else:
			checkpoint_path = checkpoint_callback.best_model_path

			print(f"\n\nBest model saved on path {checkpoint_path}\n\n")
			wandb.log({"bestmodel/step": checkpoint_path.split("step=")[1].split('.ckpt')[0]})

		#### Compute metrics for the best model
		model.log_test_key = 'bestmodel_train'
		trainer.test(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)

		model.log_test_key = 'bestmodel_valid'
		trainer.test(model, dataloaders=data_module.val_dataloader()[0], ckpt_path=checkpoint_path)

		model.log_test_key = 'bestmodel_test'
		trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)

	
	wandb.finish()

	print("\nExiting from train function..")
	

if __name__ == "__main__":
    main()