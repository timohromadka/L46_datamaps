import os
import glob

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append('..')
from utils.dataset_utils import NUM_CHANNELS, NUM_CLASSES

from efficientnet_pytorch import EfficientNet
from torchmetrics import Precision, Recall, F1Score, Accuracy

# TODO:
# Add new architectures?
# - EfficientNet
# - VisionTransformer
# - VGG


class TrainingLightningModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES[args.dataset])
        self.precision = Precision(task='multiclass', num_classes=NUM_CLASSES[args.dataset], average='macro')
        self.recall = Recall(task='multiclass', num_classes=NUM_CLASSES[args.dataset], average='macro')
        self.f1 = F1Score(task='multiclass', num_classes=NUM_CLASSES[args.dataset], average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        
        # TODO: make efficient test metric logging
        # The only workaround I could find...
        # accumulate all and calculate each step since 'on_test_epoch_end' is bugged
        if batch_idx == 0:
            self.all_preds = preds
            self.all_labels = y
        else:
            self.all_preds = torch.cat((self.all_preds, preds), dim=0)
            self.all_labels = torch.cat((self.all_labels, y), dim=0)
            
        acc = self.accuracy(self.all_preds, self.all_labels)
        precision = self.precision(self.all_preds, self.all_labels)
        recall = self.recall(self.all_preds, self.all_labels)
        f1 = self.f1(self.all_preds, self.all_labels)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', recall, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_epoch=True, prog_bar=True, logger=True)

        

    def configure_optimizers(self):
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError("Unsupported optimizer type")

        return optimizer
    
    def on_train_end(self):
        super().on_train_end()

        checkpoint_path = self.trainer.checkpoint_callback.best_model_path
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            
            # keep track of the model and model_size so that it can be automatically configured given JUST the run path
            model_config = {
                'model_type': self.args.model,
                'model_size': self.args.model_size,
                'val_split_seed': self.args.val_split_seed
            }
            checkpoint['config'] = model_config
            torch.save(checkpoint, checkpoint_path)

    

# TODO
# fix CNN model initialization with super class
class SmallCNNModel(TrainingLightningModule):
    def __init__(self, args):
        self._create_architecture(args)
        super().__init__(self, args)

    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


class MediumCNNModel(TrainingLightningModule):
    def __init__(self, args):
        self._create_architecture(args)
        super().__init__(self, args) 

    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



class LargeCNNModel(TrainingLightningModule):
    def __init__(self, args):
        self._create_architecture(args)
        super().__init__(self, args) 
        
    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SmallResNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        model = models.resnet18(pretrained=args.pretrained)
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)

class MediumResNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        model = models.resnet34(pretrained=args.pretrained)
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def forward(self, x):
        return self.model(x)

    
class LargeResNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        model = models.resnet50(pretrained=args.pretrained)
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)
    

class EfficientNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        if args.model_size == 'small':
            efficientnet_model_size = '0'
        elif args.model_size == 'medium':
            efficientnet_model_size = '2'
        elif args.model_size == 'large':
            efficientnet_model_size = '4'
        else:
            raise ValueError("Invalid model size specified. Choose from 'small', 'medium', or 'large'.")
        efficientnet_size = f'efficientnet-b{efficientnet_model_size}'

        model = EfficientNet.from_pretrained(efficientnet_size, num_classes=num_classes)

        # Modify the first convolutional layer if the number of input channels is not 3
        if num_channels != 3:
            original_conv = model._conv_stem
            model._conv_stem = nn.Conv2d(num_channels, original_conv.out_channels, 
                                         kernel_size=original_conv.kernel_size, 
                                         stride=original_conv.stride, 
                                         padding=original_conv.padding, 
                                         bias=False)

        return model


def get_model(args):
    if args.model == "cnn":
        if args.model_size == "small":
            return SmallCNNModel(args)
        elif args.model_size == "medium":
            return MediumCNNModel(args)
        elif args.model_size == "large":
            return LargeCNNModel(args)
        
    elif args.model == "resnet":
        if args.model_size == "small":
            return SmallResNetModel(args)
        elif args.model_size == "medium":
            return MediumResNetModel(args)
        elif args.model_size == "large":
            return LargeResNetModel(args)
        
    elif args.model == 'efficientnet':
        return EfficientNetModel(args)

    else:
        raise ValueError(f"Invalid model type: {args.model}. Expected 'cnn' or 'resnet'.")

def load_checkpoint(teacher_run_name, args):
    """
    Load a model checkpoint from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - model checkpoint.
    """
    teacher_model_path = os.path.join(args.checkpoint_dir, teacher_run_name)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    checkpoint_path = checkpoint_files[0]

    # Load the checkpoint to access the configuration
    checkpoint = torch.load(checkpoint_path)
    
    return checkpoint
        
def load_model_from_run_name(teacher_run_name, args):
    """
    Load a model from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - Loaded model.
    """

    # Load the checkpoint to access the configuration
    checkpoint = load_checkpoint(teacher_run_name, args)
    config = checkpoint.get('config')

    if not config:
        raise ValueError(f"No config found in checkpoint at {teacher_run_name}")

    # Determine the model class based on the configuration
    model_type = config.get('model_type')
    model_size = config.get('model_size')

    if model_type == "cnn":
        model_class = SmallCNNModel if model_size == "small" else \
                      MediumCNNModel if model_size == "medium" else \
                      LargeCNNModel
    elif model_type == "resnet":
        model_class = SmallResNetModel if model_size == "small" else \
                      MediumResNetModel if model_size == "medium" else \
                      LargeResNetModel
    elif model_type == 'efficientnet':
        model_class = EfficientNetModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_class.load_from_checkpoint(checkpoint_path, args=args)

    return model