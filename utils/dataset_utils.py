import glob
import logging
import os

import torch
from torchvision import datasets, transforms
from torchaudio.datasets import SPEECHCOMMANDS #, URBANSOUND8K
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import Subset
from models.models import load_checkpoint
from utils.wandb_utils import get_training_dynamics_from_run_name
from utils.training_dynamic_utils import get_data_subset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')

class CustomDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

def load_val_split_seed_from_run_name(teacher_run_name, args):
    """
    Load a val_split_seed from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - Loaded val_split_seed, int.
    """
    logger.info(f'Loading validation split seed from parent run name: {teacher_run_name}')

    # Load the checkpoint to access the configuration
    checkpoint = load_checkpoint(teacher_run_name, args)
    config = checkpoint.get('config')
    val_split_seed = config.get('val_split_seed')
    
    return val_split_seed

def get_train_val_test_sets(dataset_name, val_split_seed, prev_run_name_for_dynamics, keep_full=False):
    logger.info(f'Fetching train, val, and test sets according to args. dataset_name: {dataset_name} | val_split_seed: {val_split_seed} | prev_run_name_for_dynamics: {prev_run_name_for_dynamics}')
    
    # Define transformations for image datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'speechcommands':
        train_dataset = SPEECHCOMMANDS(root='./data', subset='training')
        test_dataset = SPEECHCOMMANDS(root='./data', subset='testing')
        # TODO
        # transform to spectrograms, or keep at waveform?
        
    # elif dataset_name == 'urbansound8k':
    #     train_dataset = URBANSOUND8K(root='./data', subset='training')
    #     test_dataset = URBANSOUND8K(root='./data', subset='testing')
    #     # TODO
    #     # transform to spectrograms, or keep at waveform?
    else:
        raise ValueError("Unknown dataset")
    
    if keep_full:
        return train_dataset, test_dataset
    
    # Split the training dataset into train and validation
    local_generator = torch.Generator()
    if val_split_seed:
        local_generator.manual_seed(val_split_seed)
    else:
        if prev_run_name_for_dynamics:
            raise Exception("if using datamapped subset for training, val split seed for prev run must be provided.")
        val_split_seed = local_generator.initial_seed()
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=local_generator)
    
    
    return train_dataset, val_dataset, test_dataset

    
def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_train_val_test_sets(args.dataset, args.val_split_seed, args.prev_run_name_for_dynamics)

    # only obtain a subset if it is requested !
    if args.prev_run_name_for_dynamics and \
        (args.p_easytolearn or args.p_ambiguous or args.p_hardtolearn or args.p_variability or 
         args.p_confidence or args.p_correctness or args.p_forgetfulness or args.p_random):
            
        logger.info(f'Fetching subset of data according to given training dynamic percentages.')
        
        gold_label_probabilities, confidence, variability, correctness, forgetfulness = get_training_dynamics_from_run_name('all_datasets_training_teacher_models', 'l46_datamaps', args.prev_run_name_for_dynamics)
        selected_indices = get_data_subset(
            list(range(len(train_dataset))),
            variability,
            confidence,
            correctness,
            forgetfulness,
            p_easytolearn=args.p_easytolearn,
            p_ambiguous=args.p_ambiguous,
            p_hardtolearn=args.p_hardtolearn,
            p_variability=args.p_variability,
            selector_variability=args.selector_variability, 
            p_confidence=args.p_confidence,
            selector_confidence=args.selector_confidence,  
            p_correctness=args.p_correctness,
            selector_correctness=args.selector_correctness,  
            p_forgetfulness=args.p_forgetfulness,
            selector_forgetfulness=args.selector_forgetfulness,
            p_random=args.p_random
        )

        while len(selected_indices) % args.train_batch_size < 10 and len(selected_indices) % args.train_batch_size != 0:
            selected_indices.pop() # this resolves a weird edge case bug where the last batch is too small and it causes errors.

        train_dataset = Subset(train_dataset, selected_indices)

    logger.info(f'Fetching dataloaders.')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # Unshuffled required for datamap_callback later on
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=False)

    return train_loader, train_unshuffled_loader, val_loader, test_loader

def preprocess_cifar10(dataset):
    return dataset

def preprocess_cifar100(dataset):
    return dataset

def preprocess_mnist(dataset):
    return dataset

def preprocess_speechcommands(dataset):
    return dataset

def preprocess_urbansounds8k(dataset):
    return dataset
