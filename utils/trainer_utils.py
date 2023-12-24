import os
import wandb

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from .knowledge_distillation_utils import kd_training_step, lsp_training_step, label_smoothed_nll_loss

import sys
sys.path.append("..")
from callbacks.training_dynamics_callback import DataMapLightningCallback
from models.models import get_model, load_model_from_run_name

def train_model(args, data_module, train_unshuffled_loader, wandb_logger=None):
    """
    Return 
    - Pytorch Lightning Trainer
    - checkpoint callback
    - datamap callback
    """
    
    # ========================
    # setup
    # ========================
    pl.seed_everything(args.seed, workers=True)
    
    model = get_model(args)

    mode_metric = 'max' if args.metric_model_selection == 'balanced_accuracy' else 'min'
    
    # ========================
    # bacllbacks
    # ========================
    checkpoint_callback = ModelCheckpoint(
        monitor=args.metric_model_selection,
        mode=mode_metric,
        save_top_k=args.save_top_k,
        verbose=True,
        dirpath=os.path.join(args.checkpoint_dir, args.wandb_run_name),
        filename='{epoch}_{step}_{valid_loss:.5f}'
    )
    callbacks = [checkpoint_callback, RichProgressBar()]


    if args.track_training_dynamics:
        datamap_callback = DataMapLightningCallback(
            train_unshuffled_loader,
            outputs_to_probabilities=lambda x, dim: F.softmax(x, dim),
            run_name=args.wandb_run_name,
            training_dynamics_dir=args.training_dynamics_dir
        )
        callbacks.append(datamap_callback)
        
        
    if args.patience_early_stopping and not args.train_on_full_data:
        callbacks.append(EarlyStopping(
            monitor=args.metric_model_selection,
            mode=mode_metric,
            patience=args.patience_early_stopping,
        ))
        
        
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # ========================
    # Knowledge Distillation
    # ========================
    if args.distil_experiment:
        teacher_model = load_model_from_run_name(args.teacher_model_run, args)
        teacher_model.eval()

        if args.knowledge_distillation_loss == 'KD':
            model.training_step = lambda batch, batch_idx: kd_training_step(
                batch, batch_idx, model, teacher_model, args.distillation_temp
            )
        elif args.knowledge_distillation_loss == 'LSP':
            model.training_step = lambda batch, batch_idx: lsp_training_step(
                batch, batch_idx, model, args.label_smoothing
            )

    # ========================
    # Run training and testing
    # ========================
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # max_steps=args.max_steps, # let's stick with epochs
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
