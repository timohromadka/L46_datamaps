import torch
from torchvision import datasets, transforms
from torchaudio.datasets import SPEECHCOMMANDS #, URBANSOUND8K
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from utils.wandb_utils import get_training_dynamics_from_run_name
from utils.training_dynamic_utils import get_data_subset
from torch.utils.data.dataset import Subset


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'speechcommands': 10,
    'urbansound8k': 10
}

NUM_CHANNELS = {
    'cifar10': 3,
    'cifar100': 3,
    'mnist': 1,
    'speechcommands': 3,
    'urbansound8k': 3
}

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


def get_dataloaders(args):
    # Define transformations for image datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif args.dataset == 'speechcommands':
        train_dataset = SPEECHCOMMANDS(root='./data', subset='training')
        test_dataset = SPEECHCOMMANDS(root='./data', subset='testing')
        # TODO
        # transform to spectrograms, or keep at waveform?
        
    # elif args.dataset == 'urbansound8k':
    #     train_dataset = URBANSOUND8K(root='./data', subset='training')
    #     test_dataset = URBANSOUND8K(root='./data', subset='testing')
    #     # TODO
    #     # transform to spectrograms, or keep at waveform?
    else:
        raise ValueError("Unknown dataset")
    
    if args.prev_run_name_for_dynamics:
        # get subset from dataset using previous run dynamics
        gold_label_probabilities, confidence, variability, correctness, forgetfulness = get_training_dynamics_from_run_name(args.wandb_project_name, 'l46_datamaps', args.prev_run_name_for_dynamics)
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
            selector_forgetfulness=args.selector_forgetfulness  
        )

        # Create a Subset object for the selected indices
        train_dataset = Subset(train_dataset, selected_indices)
    # Split the training dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) # required for datamap_callback later on
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