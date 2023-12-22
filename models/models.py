import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import torch.nn as nn
import torchvision.models as models

from torchmetrics import Precision, Recall, F1Score, Accuracy


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'speechcommands': 10,
    'urbansound8k': 10
}


class TrainingLightningModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES[args.dataset])
        self.precision = Precision(task='multiclass', num_classes=NUM_CLASSES[args.dataset])
        self.recall = Recall(task='multiclass', num_classes=NUM_CLASSES[args.dataset])
        self.f1 = F1Score(task='multiclass', num_classes=NUM_CLASSES[args.dataset])

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

        acc = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # # NOTE: https://github.com/Lightning-AI/pytorch-lightning/pull/16520
    # # migrated to new function names
    # # have outputs as a parameter (should be removed soon anyways)
    # def test_epoch_end(self, outputs):
    #     self.log('test_precision', self.precision.compute(), prog_bar=True, logger=True)
    #     self.log('test_recall', self.recall.compute(), prog_bar=True, logger=True)
    #     self.log('test_f1', self.f1.compute(), prog_bar=True, logger=True)

    #     self.precision.reset()
    #     self.recall.reset()
    #     self.f1.reset()
    #     self.accuracy.reset()
        
        

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

    
    
class SmallCNNModel(TrainingLightningModule):
    def __init__(self, args):
        self._create_architecture(args)
        super().__init__(self, args)

    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
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
        super().__init__(self, args)  # Pass 'self' as the model to the parent class

    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
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
        super().__init__(self, args)  # Pass 'self' as the model to the parent class

    def _create_architecture(self, args):
        num_classes = NUM_CLASSES[args.dataset]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
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
        model = models.resnet18(pretrained=args.pretrained)
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
        model = models.resnet34(pretrained=args.pretrained)
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
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)


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

    else:
        raise ValueError(f"Invalid model type: {args.model}. Expected 'cnn' or 'resnet'.")

        
