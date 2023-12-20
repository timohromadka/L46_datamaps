import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Training Dynamics Guided Knowledge Distillation.')

"""
datasets

-- IMAGE --
- cifar-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
- cifar-100 (https://www.cs.toronto.edu/~kriz/cifar.html)
- mnist (https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

-- AUDIO --
- SpeechCommands (https://paperswithcode.com/dataset/speech-commands)
- UrbanSound8k (https://urbansounddataset.weebly.com/urbansound8k.html)
"""

"""
models

-- IMAGE --

-- AUDIO --

"""
# Dataset
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'mnist', 'speechcommands', 'urbansound8k'])
parser.add_argument('--small_dataset', action='store_true', help='(for debugging) True if we wish to use experiment with only a small sample of data from training set.')

# Model 
# Note, student models are to also be defined here
parser.add_argument('--model', type=str, required=True, choices=['cnn', 'resnet'])
parser.add_argument('--model_size', type=str, required=True, choices=['small', 'medium', 'large'])


# Training Configuration
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'])

# PyTorch Lightning Specific
parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
parser.add_argument('--accelerator', type=str, default='GPU')
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--gradient_clip_val', type=float, default=0.5)

# Weights & Biases (wandb) Integration
parser.add_argument('--wandb_project', type=str, default='L46_datamaps')
parser.add_argument('--wandb_run_name', type=str, default=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
parser.add_argument('--wandb_log_freq', type=int, default=10)

# Knowledge Distillation
parser.add_argument('--teacher_model_path', type=str, required=True)
parser.add_argument('--distillation_temp', type=float, default=2)
parser.add_argument('--p_hardtolearn', type=int)
parser.add_argument('--p_ambiguous', type=int)
parser.add_argument('--p_easytolearn', type=int)
parser.add_argument('--selection_from_low', action='store_true', help='Enables selection from the subset with the lowest values.')
parser.add_argument('--p_confidence', type=int)
parser.add_argument('--p_variability', type=int)
parser.add_argument('--p_correctness', type=int)
parser.add_argument('--p_forgetfulness', type=int)

# Miscellaneous
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--logging_interval', type=int, default=50)
parser.add_argument('--checkpoint_freq', type=int, default=5)

####### Wandb logging
parser.add_argument('--group', type=str, help="Group runs in wand")
parser.add_argument('--job_type', type=str, help="Job type for wand")
parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model',
                    help='True for storing the model checkpoints in wandb')
parser.set_defaults(wandb_log_model=False)
parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb',
                    help='True if you dont want to crete wandb logs.')
parser.set_defaults(disable_wandb=False)

