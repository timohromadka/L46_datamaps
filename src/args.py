import argparse
from datetime import datetime

# ======================================
# arg checking utils
# ======================================
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0,1]")
    return x
# ======================================

parser = argparse.ArgumentParser(description='Training Dynamics Guided Knowledge Distillation.')

# ======================================
# TODO
# - rearrange these argument into actually meaningful groups
# ======================================

# Dataset
parser.add_argument('--dataset', type=str, required=False, choices=['cifar10', 'cifar100', 'mnist', 'speechcommands', 'urbansound8k', 'pass_via_subset_argument'])
# parser.add_argument('--small_dataset', action='store_true', help='(for debugging) True if using a small sample of data from the training set.')

# Model 
parser.add_argument('--model', type=str, required=True, choices=['resnet', 'efficientnet', 'visualtransformer', 'vgg', 'mobilenet'])
parser.add_argument('--model_size', type=str, required=True, choices=['small', 'medium', 'large'])
parser.add_argument('--pretrained', action='store_true', help='If True, use pretrained weights (usually pretrained on ImageNet).')
parser.add_argument('--pretrained_from_github', action='store_true', help='If True, will use an already high-performing trained model for the specific dataset as defined in --dataset.')

# Training Configuration
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs for training.')
parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum number of training steps (batches) for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'], help='Optimizer for training.')
parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--validation_batch_size', type=int, default=32, help='Batch size for validation.')
parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Gradient clipping value to prevent exploding gradients.')
parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints/', help='Directory to save model checkpoints.')
parser.add_argument('--force_full_epoch_training', action='store_true', help='If True, then training will continue for the specified amount of epochs regardless.')
# PyTorch Lightning Specific
parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Precision of training (32 or 16 for mixed precision).')
parser.add_argument('--accelerator', type=str, default='gpu', help='Type of accelerator to use ("gpu" or "cpu").')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
parser.add_argument('--deterministic', action='store_true', help='Ensures reproducibility. May impact performance.')

# General Configuration
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--data_augmentation', type=bool, default=True, help='Whether to use data augmentation.')
parser.add_argument('--logging_interval', type=int, default=100, help='Interval for logging training metrics.')
# parser.add_argument('--checkpoint_freq', type=int, default=3, help='Frequency of saving top-k model checkpoints.')
parser.add_argument('--save_top_k', type=int, default=1, help='Select k-best model checkpoints to save for each run.')
parser.add_argument('--test_only', action='store_true', help='If True, will only run testing, no training or validation will be performed.')


# Validation
parser.add_argument('--metric_model_selection', type=str, default='val_loss',
                    choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy', 'accuracy', 'lr-Adam', 'train_loss', 'train_loss_step', 'train_acc', 'train_acc_step', 'val_loss', 'val_acc'], help='Metric used for model selection.')
parser.add_argument('--patience_early_stopping', type=int, default=3,
                    help='Set number of checks (set by *val_check_interval*) to do early stopping. Minimum training duration: args.val_check_interval * args.patience_early_stopping epochs')
parser.add_argument('--val_check_interval', type=float, default=1.0, 
                    help='Number of steps at which to check the validation. If set to 1.0, will simply perform the default behaviour of an entire batch before validation.')
parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
                    help='Train on the full data (train + validation), leaving only `--test_split` for testing.')
parser.add_argument('--overfit_batches', type=int, default=0, help='PyTorch Lightning trick to pick only N batches and iteratively overfit on them. Useful for debugging. Default set to 0, i.e. normal behaviour.')

# Training Dynamics
parser.add_argument('--track_training_dynamics', action='store_true', help='If True, the current run will track training dynamics at the end of each epoch.')
parser.add_argument('--training_dynamics_dir', type=str, default='training_dynamics')
parser.add_argument('--prev_run_name_for_dynamics', type=str, default='',help='The wandb run_name to use to fetch training_dynamics from previous run. Default=None if not training using subset')
parser.add_argument('--val_split_seed', type=int, required=True, help='Random seed for train-val split. Required for datamapped subset selection. If not provided, will be sourced from teacher in case of student or will be randomly generated in case of teacher.')

# Knowledge Distillation
parser.add_argument('--distil_experiment', action='store_true', help='If True, the current run will now be knowledge distillation.')
parser.add_argument('--teacher_model_run', type=str, default='')
parser.add_argument('--distillation_temp', type=float, default=1)
parser.add_argument('--hard_label_loss', type=str, default='cross_entropy', choices=['cross_entropy']) # add more if needed
parser.add_argument('--knowledge_distillation_loss', type=str, default='KD', choices=['KD'], help='Type of knowledge distillation loss to use')
parser.add_argument('--knowledge_distillation_loss_alpha', type=restricted_float, default=0.5, help='Specify the alpha value, as a float, of how much the knowledge distillation loss contributes. Should be in [0,1]')
parser.add_argument('--knowledge_distillation_label_smoothing', type=restricted_float, default=0.1, help='Specify the label smoothing value, as a float. Should be in [0,1]')
parser.add_argument('--p_hardtolearn', type=float, default=0.0)
parser.add_argument('--p_ambiguous', type=float, default=0.0)
parser.add_argument('--p_easytolearn', type=float, default=0.0)
parser.add_argument('--selection_from_low', action='store_true', help='Enables selection from the subset with the lowest values.')
parser.add_argument('--p_confidence', type=float, default=0.0)
parser.add_argument('--p_variability', type=float, default=0.0)
parser.add_argument('--p_correctness', type=float, default=0.0)
parser.add_argument('--p_forgetfulness', type=float, default=0.0)
parser.add_argument('--p_random', type=float, default=0.0)
parser.add_argument('--selector_variability', type=str, choices=['top', 'bottom'], default='top')
parser.add_argument('--selector_confidence', type=str, choices=['top', 'bottom'], default='top')
parser.add_argument('--selector_correctness', type=str, choices=['top', 'bottom'], default='top')
parser.add_argument('--selector_forgetfulness', type=str, choices=['top', 'bottom'], default='top')
parser.add_argument('--subset_argument', type=str, help='Arguments related to data subsets.')


# Weights & Biases (wandb) Integration
parser.add_argument('--wandb_project_name', type=str, default='L46_datamaps')
parser.add_argument('--wandb_run_name', type=str, default=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
parser.add_argument('--wandb_log_freq', type=int, default=10)
parser.add_argument('--group', type=str, help="Group runs in wand")
parser.add_argument('--job_type', type=str, help="Job type for wand")
parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model', help='True for storing the model checkpoints in wandb')
parser.set_defaults(wandb_log_model=False)
parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb', help='True if you dont want to create wandb logs.')
parser.set_defaults(disable_wandb=False)

def apply_subset_arguments(subset_args_str, args):

    # Proceed only if the string is not empty
    if subset_args_str and subset_args_str is not None:
        # Split the subset argument string into individual arguments
        # Trim the string to remove any leading/trailing whitespace
        subset_args_str = subset_args_str.strip()
        subset_args = subset_args_str.split()
        
        # Iterate over the subset arguments and update the args Namespace
        i = 0
        while i < len(subset_args):
            arg = subset_args[i]
            # Ensure that it starts with '--'
            if arg.startswith("--"):
                key = arg[2:]  # Remove '--' prefix to match the args keys
                value = subset_args[i + 1]
                # Update the args Namespace if the attribute exists
                if hasattr(args, key):
                    # Convert value to the right type based on the existing attribute
                    attr_type = type(getattr(args, key))
                    setattr(args, key, attr_type(value))
                i += 2  # Move to the next argument
            else:
                raise ValueError(f"Expected an argument starting with '--', found: {arg}")


