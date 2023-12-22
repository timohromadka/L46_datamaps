import os
import wandb

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append("..")
from callbacks.training_dynamics_callback import DataMapLightningCallback


def train_model(args, model, data_module, train_unshuffled_loader, wandb_logger=None):
    """
    Return 
    - Pytorch Lightning Trainer
    - checkpoint callback
    - datamap callback
    """
    pl.seed_everything(args.seed, workers=True)

    mode_metric = 'max' if args.metric_model_selection == 'balanced_accuracy' else 'min'
    
    checkpoint_callback = ModelCheckpoint(
        #monitor=f'valid/{args.metric_model_selection}',
        monitor=f'{args.metric_model_selection}',
        mode=mode_metric,
        save_last=True,
        verbose=True,
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{step}-{valid_loss:.2f}',
        save_top_k=args.checkpoint_freq
    )

    datamap_callback = DataMapLightningCallback(
        train_unshuffled_loader,
        args.model,
        args.dataset,
        outputs_to_probabilities=lambda x: F.softmax(x[0]),
        sparse_labels=True,
    )
    callbacks = [checkpoint_callback, RichProgressBar(), datamap_callback]

    if args.patience_early_stopping and not args.train_on_full_data:
        callbacks.append(EarlyStopping(
            # monitor=f'valid/{args.metric_model_selection}',
            monitor=f'{args.metric_model_selection}',
            mode=mode_metric,
            patience=args.patience_early_stopping,
        ))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.num_gpus,
        precision=args.precision,
        # detect_anomaly=True,
        detect_anomaly=False,
        overfit_batches=args.overfit_batches,
        deterministic=args.deterministic,
    )
 
    trainer.fit(model, data_module)
    
    trainer.test(model, dataloaders=data_module.test_dataloader())

    return trainer, checkpoint_callback, datamap_callback