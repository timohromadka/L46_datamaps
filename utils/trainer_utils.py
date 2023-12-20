import os
import wandb

import torch
import torch.nn.functional as F
import pytorch_lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from ..callbacks.training_dynamics_callback import DataMapLightningCallback


def train_model(args, model, data_module, wandb_logger=None):
	"""
	Return 
	- Pytorch Lightning Trainer
	- checkpoint callback
	"""

	##### Train
	if args.saved_checkpoint_name:
		wandb_artifact_path = f'th716/low-data/{args.saved_checkpoint_name}'
		print(f"\nDownloading artifact: {wandb_artifact_path}...")

		artifact = wandb.use_artifact(wandb_artifact_path, type='model')
		artifact_dir = artifact.download()
		model_checkpoint = torch.load(os.path.join(artifact_dir, 'model.ckpt'))
		weights = model_checkpoint['state_dict']
		print("Artifact downloaded")

		if args.load_model_weights:
			print(f"\nLoading pretrained weights into model...")
			missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
			print(f"Missing keys: \n")
			print(missing_keys)

			print(f"Unexpected keys: \n")
			print(unexpected_keys)

	mode_metric = 'max' if args.metric_model_selection=='balanced_accuracy' else 'min'
	checkpoint_callback = ModelCheckpoint(
		monitor=f'valid/{args.metric_model_selection}',
		mode=mode_metric,
		save_last=True,
		verbose=True
	)
 
    #  return DataMapCallback(train_unshuffled.batch(batch_size),
    #                        train_unshuffled_raw,
    #                        model_name,
    #                        dataset_name,
    #                         outputs_to_probabilities=lambda x: tf.nn.softmax(x[0]), # Model outputs a tuple, where the logits are at the first index
    #                         sparse_labels=True,
    #                         gold_labels_probabilities=gold_labels_probabilities)
 
	datamap_callback = DataMapLightningCallback(
		data_module,
		args.model_name,
		args.dataset_name,
  		outputs_to_probabilities=lambda x: F.softmax(x[0]), # Model outputs a tuple, where the logits are at the first index
		sparse_labels=True, # conversion to one hot encoding necessary
	)
	callbacks = [checkpoint_callback, RichProgressBar(), datamap_callback]

	if args.patience_early_stopping and args.train_on_full_data==False:
		callbacks.append(EarlyStopping(
			monitor=f'valid/{args.metric_model_selection}',
			mode=mode_metric,
			patience=args.patience_early_stopping,
		))
	callbacks.append(LearningRateMonitor(logging_interval='step'))

	pl.seed_everything(args.seed_training, workers=True)
	trainer = pl.Trainer(
		# Training
		max_steps=args.max_steps,
		gradient_clip_val=2.5,

		# logging
		logger=wandb_logger,
		log_every_n_steps = 1,
		val_check_interval = args.val_check_interval,
		callbacks = callbacks,

		# miscellaneous
		accelerator="auto",
		devices="auto",
		detect_anomaly=True,
		overfit_batches=args.overfit_batches,
		deterministic=args.deterministic,
	)
	# train
	trainer.fit(model, data_module)
	
	return trainer, checkpoint_callback, datamap_callback