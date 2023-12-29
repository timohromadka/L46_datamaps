import copy
import logging
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
from utils.constants import NUM_CHANNELS, NUM_CLASSES

from efficientnet_pytorch import EfficientNet
from torchmetrics import Precision, Recall, F1Score, Accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models/models.py')

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
        if self.args.model == 'visualtransformer' and self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(0.9, 0.98),
                eps=1e-9
            )
        elif self.args.optimizer == 'Adam':
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
                'val_split_seed': self.args.val_split_seed,
                'pretrained_from_github': self.args.pretrained_from_github,
                'pretrained': self.args.pretrained
            }
            checkpoint['config'] = model_config
            torch.save(checkpoint, checkpoint_path)


    
class VGGModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]

        if args.pretrained_from_github:
            if args.model_size == 'small':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_vgg11_bn", pretrained=True)
            elif args.model_size == 'medium':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_vgg13_bn", pretrained=True)
            elif args.model_size == 'large':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_vgg16_bn", pretrained=True)
            else:
                raise ValueError("Invalid model size. Expected 'small', 'medium', or 'large'.")
            return model

        if args.model_size == 'small':
            model = models.vgg11(pretrained=args.pretrained)
        elif args.model_size == 'medium':
            model = models.vgg13(pretrained=args.pretrained)
        elif args.model_size == 'large':
            model = models.vgg16(pretrained=args.pretrained)
        else:
            raise ValueError("Invalid model size. Expected 'small', 'medium', or 'large'.")

        # Modify the first convolutional layer if the number of input channels is not 3
        if num_channels != 3:
            first_conv_layer = model.features[0]
            model.features[0] = nn.Conv2d(num_channels, first_conv_layer.out_channels,
                                               kernel_size=first_conv_layer.kernel_size,
                                               stride=first_conv_layer.stride,
                                               padding=first_conv_layer.padding)

        # Replace the classifier with a new one matching the number of classes
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        
        return model

    def forward(self, x):
        return self.model(x)


class ResNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        
        if args.pretrained_from_github:
            if args.model_size == 'small':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_resnet20", pretrained=True)
            elif args.model_size == 'medium':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_resnet32", pretrained=True)
            elif args.model_size == 'large':
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{args.dataset}_resnet44", pretrained=True)
            else:
                raise ValueError("Invalid model size. Expected 'small', 'medium', or 'large'.")
            return model

        if args.model_size == 'small':
            model = models.resnet18(pretrained=args.pretrained)
        elif args.model_size == 'medium':
            model = models.resnet34(pretrained=args.pretrained)
        elif args.model_size == 'large':
            model = models.resnet50(pretrained=args.pretrained)
        else:
            raise ValueError("Invalid model size. Expected 'small', 'medium', or 'large'.")
    
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)

# class SmallResNetModel(TrainingLightningModule):
#     def __init__(self, args):
#         model = self._create_model(args)  
#         super().__init__(model, args)

#     def _create_model(self, args):
#         num_classes = NUM_CLASSES[args.dataset]
#         num_channels = NUM_CHANNELS[args.dataset]
#         model = models.resnet18(pretrained=args.pretrained)
#         if num_channels != 3:
#             model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model

#     def forward(self, x):
#         return self.model(x)

# class MediumResNetModel(TrainingLightningModule):
#     def __init__(self, args):
#         model = self._create_model(args)  
#         super().__init__(model, args)

#     def _create_model(self, args):
#         num_classes = NUM_CLASSES[args.dataset]
#         num_channels = NUM_CHANNELS[args.dataset]
#         model = models.resnet34(pretrained=args.pretrained)
#         if num_channels != 3:
#             model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model
    
#     def forward(self, x):
#         return self.model(x)

    
# class LargeResNetModel(TrainingLightningModule):
#     def __init__(self, args):
#         model = self._create_model(args)  
#         super().__init__(model, args)

#     def _create_model(self, args):
#         num_classes = NUM_CLASSES[args.dataset]
#         num_channels = NUM_CHANNELS[args.dataset]
#         model = models.resnet50(pretrained=args.pretrained)
#         if num_channels != 3:
#             model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model

#     def forward(self, x):
#         return self.model(x)
    

class EfficientNetModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        num_channels = NUM_CHANNELS[args.dataset]
        
        if args.pretrained_from_github:
            raise ValueError("Loading pretrained models from GitHub with an efficientnet is not allowed.")

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

class ViTModel(TrainingLightningModule):
    def __init__(self, args):
        model = self._create_model(args)  
        super().__init__(model, args)

    def _create_model(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        
        if args.dataset == 'cifar10':
            image_size = 32
        elif args.dataset == 'cifar100':
            image_size = 32
        elif args.dataset == 'mnist':
            image_size = 28
        else:
            raise Exception(f"Dataset: <{args.dataset}> not supported for ViT")
        
        if args.model_size == 'small':
            model = models.vit_b_32(pretrained=args.pretrained, num_classes=num_classes, image_size=image_size)
        elif args.model_size == 'large':
            model = models.vit_l_32(pretrained=args.pretrained, num_classes=num_classes, image_size=image_size)
        elif args.model_size == 'medium':
            raise ValueError("Medium size is not supported. Choose from 'small' or 'large' for visualtransformer.")
        else:
            raise ValueError("Invalid model size specified. Choose from 'small' or 'large' for visualtransformer.")


        return model
    
def get_model(args):
    logger.info(f'Fetching model. args.model {args.model} | args.model_size: {args.model_size}')

    if args.model == "resnet":
        return ResNetModel(args)
        
    elif args.model == 'efficientnet':
        return EfficientNetModel(args)
    
    elif args.model == 'vgg':
        return VGGModel(args)
    
    elif args.model == 'visualtransformer':
        return ViTModel(args)
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
    logger.info(f'Loading model checkpoint from run name: {teacher_run_name}')
    
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
    logger.info(f'Loading, configuring, and initializing teacher model from checkpoint using teacher run name: {teacher_run_name}')
    
    teacher_model_path = os.path.join(args.checkpoint_dir, teacher_run_name)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    # There should only be one, if not, we only grab the first one for simplicity
    checkpoint_path = checkpoint_files[0]

    # Load the checkpoint to access the configuration
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint.get('config')

    if not config:
        raise ValueError(f"No config found in checkpoint at {checkpoint_path}")

    # Determine the model class based on the configuration
    model_type = config.get('model_type')
    model_size = config.get('model_size')
    pretrained_from_github = config.get('pretrained_from_github')
    pretrained = config.get('pretrained')
    
    # Create a new copy of args for the teacher model
    teacher_args = copy.deepcopy(args)
    teacher_args.model_size = model_size
    teacher_args.pretrained_from_github = pretrained_from_github
    teacher_args.pretrained = pretrained

    if model_type == "resnet":
        model = ResNetModel
    elif model_type == 'efficientnet':
        model = EfficientNetModel
    elif args.model == 'vgg':
        return VGGModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.load_from_checkpoint(checkpoint_path, args=teacher_args)

    return model