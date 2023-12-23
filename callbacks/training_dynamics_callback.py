import os
import json
import numpy as np
import wandb

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

# TODO
# - fix glitchy contiguous tqdm progress bar?
class DataMapLightningCallback(Callback):
    def __init__(self, dataloader, outputs_to_probabilities=lambda x, dim: F.softmax(x, dim), gold_labels_probabilities=None, run_name='default_run_name'):
        self.dataloader = dataloader
        self.outputs_to_probabilities = outputs_to_probabilities
        self.gold_labels_probabilities = gold_labels_probabilities
        self.run_name = run_name
        self.training_dynamics = {}
        self.correctness_across_epochs = [[] for _ in range(len(dataloader.dataset))]
        
    def convert_numpy(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return obj

    def on_train_epoch_end(self, trainer, pl_module):
        print(f'\nEpoch has ended! Now calculating gold labels probabilities.\n')

        gold_label_probabilities = []
        correctness_this_epoch = [] 
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.dataloader), desc='Calculating gold label probabilities'):
                # Predictions
                x, y = batch
                x = x.to(pl_module.device) # lightning will not handle this automatically
                probabilities = pl_module(x)
                probabilities = self.outputs_to_probabilities(probabilities, dim=1)

                # Correctness calculation (must be done here due to labels needing to be known for argmax calculation)
                # TODO: MAKE FASTER
                predicted_labels = torch.argmax(probabilities, dim=1)
                correct_predictions = (predicted_labels == y.to('cuda')).cpu().numpy()

                for idx, correct in enumerate(correct_predictions):
                    data_point_index = batch_idx * self.dataloader.batch_size + idx
                    self.correctness_across_epochs[data_point_index].append(correct)

                # Retrieve just the gold label probabilities
                batch_gold_label_probabilities = probabilities[torch.arange(probabilities.shape[0]), y].cpu().numpy()
                gold_label_probabilities.append(batch_gold_label_probabilities)

        # reshape to by a vertical stack (cols = epochs)
        gold_label_probabilities = np.concatenate(gold_label_probabilities)
        if self.gold_labels_probabilities is None:
            self.gold_labels_probabilities = gold_label_probabilities[..., None]
        else:
            self.gold_labels_probabilities = np.concatenate([self.gold_labels_probabilities, gold_label_probabilities[..., None]], axis=-1)

    def on_train_end(self, trainer, pl_module):
        print(f'\nTraining has ended! Preparing and uploading training dynamics to WandB.\n')
        
        for idx, cur_gold_label_probs in tqdm(enumerate(self.gold_labels_probabilities), desc='Calculating training dynamics from gold labels probabilities.'):
            self.training_dynamics[int(idx)] = {
                'gold_label_probs': self.convert_numpy(cur_gold_label_probs),
                'confidence': self.convert_numpy(self.confidence(cur_gold_label_probs)),
                'variability': self.convert_numpy(self.variability(cur_gold_label_probs)),
                'correctness': np.mean(self.correctness_across_epochs[idx]),
                'forgetfulness': self.convert_numpy(self.forgetfulness(cur_gold_label_probs))
            }
            
        # First, save to json locally
        json_file_name = f'{self.run_name}_training_dynamics.json'
        with open(json_file_name, 'w') as file:
            json.dump(self.training_dynamics, file, ensure_ascii=False, indent=4)

        # Then, upload saved json to wandb
        wandb.save(json_file_name)

    @staticmethod
    def confidence(gold_label_probs):
        return np.mean(gold_label_probs)

    @staticmethod
    def variability(gold_label_probs):
        return np.std(gold_label_probs)

    @staticmethod
    def forgetfulness(gold_label_probs):
        epochs = np.arange(len(gold_label_probs))
        if len(gold_label_probs) == 1:
            return 0
        m, _ = np.polyfit(epochs, gold_label_probs, 1)
        return m