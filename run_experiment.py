import logging
import os
import wandb
from typing import Dict, List, Tuple

import pytorch_lightning as pl

from src.args import parser, apply_subset_arguments
from utils.dataset_utils import get_dataloaders, CustomDataModule, load_val_split_seed_from_run_name
from utils.trainer_utils import train_model
from utils.wandb_utils import create_wandb_logger

import traceback
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Experiment Pipeline for Datamaps!')


def process_results(args):
    # If anything needs to be processed after training/testing, it should go here
    # IDEAS:
    # - perhaps some training dynamics statistics?
    return

def main():
    try:
        # ================================
        # SET UP ARGS
        # ================================
        args = parser.parse_args()
        apply_subset_arguments(args.subset_argument, args)
        # ================================
        # CONFIGURE WANDB
        # ================================
        if args.disable_wandb:
            os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(project=args.wandb_project_name, config=args)
    
        wandb_logger = create_wandb_logger(args, args.wandb_project_name)
        # wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
        wandb.run.name = args.wandb_run_name

        # ================================
        # FETCH DATASET
        # ================================
        
        if args.distil_experiment: # fetching the val split seed from the distiller
            args.val_split_seed = load_val_split_seed_from_run_name(args.teacher_model_run, args)
        train_loader, train_unshuffled_loader, val_loader, test_loader = get_dataloaders(args)
        data_module = CustomDataModule(train_loader, val_loader, test_loader)
        
        # ================================
        # UNDERGO TRAINING
        # ================================
        train_model(
            args, data_module, train_unshuffled_loader, wandb_logger
        )
        
        process_results(args)
        
    except Exception as e:
        # exit gracefully, so wandb logs errors
        print(traceback.print_exc(), file=sys.stderr)
        print(f"An error occurred: {e}\n Terminating run here.")

    finally:
        wandb.finish()
        
if __name__ == '__main__':
    main()

