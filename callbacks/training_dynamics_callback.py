import os
import json
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

class DataMapLightningCallback(Callback):
    def __init__(self, dataloader, model_name, dataset_name, outputs_to_probabilities=None, sparse_labels=False, gold_labels_probabilities=None):
        self.dataloader = dataloader
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.outputs_to_probabilities = outputs_to_probabilities
        self.sparse_labels = sparse_labels
        self.gold_labels_probabilities = gold_labels_probabilities

    def on_epoch_end(self, trainer, pl_module):
        print(f'\nEpoch has ended! Now calculating gold labels probabilities.\n')

        gold_label_probabilities = []
        pl_module.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                x, y = batch
                probabilities = pl_module(x)

                if self.outputs_to_probabilities is not None:
                    probabilities = self.outputs_to_probabilities(probabilities)

                if self.sparse_labels:
                    y = torch.nn.functional.one_hot(y, num_classes=probabilities.shape[-1])

                if y.dim() == 1:  # Binary labels
                    probabilities = probabilities.squeeze()
                    y = y.squeeze()
                    batch_gold_label_probabilities = torch.where(y == 0, 1 - probabilities, probabilities)
                elif y.dim() == 2:  # Multiclass labels
                    if not torch.all(torch.sum(y == 1, dim=-1) == 1):
                        raise ValueError('DataMapLightningCallback does not support multi-label classification')
                    batch_gold_label_probabilities = probabilities[y.bool()].cpu().numpy()
                else:
                    raise ValueError('y must be 1D for binary classification or 2D for multiclass.')

                gold_label_probabilities.append(batch_gold_label_probabilities)

        gold_label_probabilities = np.concatenate(gold_label_probabilities)
        if self.gold_labels_probabilities is None:
            self.gold_labels_probabilities = gold_label_probabilities[..., None]
        else:
            self.gold_labels_probabilities = np.concatenate([self.gold_labels_probabilities, gold_label_probabilities[..., None]], axis=-1)

        self.save_training_dynamics_to_json()

    def convert_numpy(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        return obj

    def save_training_dynamics_to_json(self):
        print(f'Saving training dynamics at the end of the epoch.')
        gold_label_probs = self.gold_labels_probabilities
        confidence = self.confidence
        variability = self.variability
        correctness = self.correctness
        forgetfulness = self.forgetfulness

        json_dict = {}
        for i, example in enumerate(tqdm(self.dataloader.dataset, desc='Saving training indexes along with training dynamics to json')):
            idx = self.convert_numpy(example["idx"])
            json_dict[idx] = {
                "gold_label_probs": self.convert_numpy(gold_label_probs[i]),
                "confidence": self.convert_numpy(confidence[i]),
                "variability": self.convert_numpy(variability[i]),
                "correctness": self.convert_numpy(correctness[i]),
                "forgetfulness": self.convert_numpy(forgetfulness[i]),
            }

        directory_path = f'training_dynamics/{self.dataset_name}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        json_file_name = f'{directory_path}/{self.model_name}.json'
        print(f'\nSaving training dynamics to {json_file_name}.')
        with open(json_file_name, 'w') as file:
            json.dump(json_dict, file, ensure_ascii=False, indent=4)

    @property
    def confidence(self):
        if self.gold_labels_probabilities is None:
            return None
        return np.mean(self.gold_labels_probabilities, axis=-1)

    @property
    def variability(self):
        if self.gold_labels_probabilities is None:
            return None
        return np.std(self.gold_labels_probabilities, axis=-1)

    @property
    def correctness(self):
        if self.gold_labels_probabilities is None:
            return None
        return np.mean(self.gold_labels_probabilities > 0.5, axis=-1)

    @property
    def forgetfulness(self):
        if self.gold_labels_probabilities is None:
            return None
        epochs = np.arange(self.gold_labels_probabilities.shape[1])

        def calculate_slope(y_values):
            if len(y_values) == 1:
                return 0
            m, _ = np.polyfit(epochs, y_values, 1)
            return m

        return np.apply_along_axis(calculate_slope, 1, self.gold_labels_probabilities)

