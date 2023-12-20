import logging
import os
import wandb
from typing import Dict, List, Tuple

import pytorch_lightning as pl

from args import get_args
from models import get_model
from dataset_utils import get_dataset
from callbacks.training_dynamics import DataMapLightningCallback
from utils.trainer_utils import train_model
from wandb_utils import create_wandb_logger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run DataMap Experimental Pipeline.')


def process_results(args):
    return

def main():
    # ================================
    # SET UP ARGS
    # ================================
    args = get_args()
    project_name = "L46_datamaps"
    
    # ================================
    # CONFIGURE WANDB
    # ================================
    wandb.init(project=project_name, config=args)
    if args.disable_wandb:
		os.environ['WANDB_MODE'] = 'disabled'
    wandb_logger = create_wandb_logger(args, project_name)
    # wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
    wandb.run_name = args.run_name

    # ================================
    # FETCH MODEL
    # ================================
    model = get_model(args)

    # ================================
    # FETCH DATASET
    # ================================
    data_module = create_data_module(args)
    #dataset = get_dataset(args) # unshuffled dataset
    
    # ================================
    # FETCH MODEL
    # ================================
    trainer, checkpoint_callback, datamap_callback = train_model(args, model, data_module, wandb_logger)
    
    process_results(args)
    
    # Done!
    wandb.finish()

if __name__ == "__main__":
    main()
