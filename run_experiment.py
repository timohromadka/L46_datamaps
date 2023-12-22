import logging
import os
import wandb
from typing import Dict, List, Tuple

import pytorch_lightning as pl

from src.args import parser
from models.models import get_model
from utils.dataset_utils import get_dataloaders, CustomDataModule
from utils.trainer_utils import train_model
from utils.wandb_utils import create_wandb_logger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Experiment Pipeline for Datamaps!')


def process_results(args):
    return

def main():
    # ================================
    # SET UP ARGS
    # ================================
    args = parser.parse_args()
    project_name = "L46_datamaps"
    
    # ================================
    # CONFIGURE WANDB
    # ================================
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=project_name, config=args)
  
    wandb_logger = create_wandb_logger(args, project_name)
    # wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
    wandb.run_name = args.wandb_run_name

    # ================================
    # FETCH MODEL
    # ================================
    model = get_model(args)

    # ================================
    # FETCH DATASET
    # ================================
    train_loader, train_unshuffled_loader, val_loader, test_loader = get_dataloaders(args)
    data_module = CustomDataModule(train_loader, val_loader, test_loader)
    
    # ================================
    # UNDERGO TRAINING
    # ================================
    trainer, checkpoint_callback, datamap_callback = train_model(
        args, model, data_module, train_unshuffled_loader, wandb_logger
    )
    
    process_results(args)
    
    # Done!
    wandb.finish()

if __name__ == "__main__":
    main()
