import torch
from torchvision import datasets, transforms
from torchaudio.datasets import SPEECHCOMMANDS, URBANSOUND8K
from torch.utils.data import DataLoader, random_split

NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'speechcommands': 10,
    'urbansound8k': 10
}

def create_data_module(args):
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
        
    elif args.dataset == 'urbansound8k':
        train_dataset = URBANSOUND8K(root='./data', subset='training')
        test_dataset = URBANSOUND8K(root='./data', subset='testing')
        # TODO
        # transform to spectrograms, or keep at waveform?
    else:
        raise ValueError("Unknown dataset")

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