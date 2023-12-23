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
    def __init__(self, dataloader, model_name, dataset_name, outputs_to_probabilities=lambda x, dim: F.softmax(x, dim), sparse_labels=False, gold_labels_probabilities=None, run_name='default_run_name'):
        self.dataloader = dataloader
        self.outputs_to_probabilities = outputs_to_probabilities
        self.sparse_labels = sparse_labels
        self.gold_labels_probabilities = gold_labels_probabilities
        self.run_name = run_name
        self.training_dynamics = {}
        

    def on_train_epoch_end(self, trainer, pl_module):
        print(f'\nEpoch has ended! Now calculating gold labels probabilities.\n')

        gold_label_probabilities = []
        pl_module.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Calculating gold label probabilities'):
                x, y = batch
                x = x.to(pl_module.device) # lightning will not handle this automatically
                probabilities = pl_module(x)

                probabilities = self.outputs_to_probabilities(probabilities, dim=1)
                
                batch_gold_label_probabilities = probabilities[torch.arange(probabilities.shape[0]), y].cpu().numpy()

                gold_label_probabilities.append(batch_gold_label_probabilities)

        gold_label_probabilities = np.concatenate(gold_label_probabilities)
        if self.gold_labels_probabilities is None:
            self.gold_labels_probabilities = gold_label_probabilities[..., None]
        else:
            self.gold_labels_probabilities = np.concatenate([self.gold_labels_probabilities, gold_label_probabilities[..., None]], axis=-1)

    def on_train_end(self, trainer, pl_module):
        print(f'\nTraining has ended! Preparing and uploading training dynamics to WandB.\n')
        
        for idx, cur_gold_label_probs in tqdm(enumerate(self.gold_labels_probabilities), desc='Calculating training dynamics from gold labels probabilities.'):
            current_dynamics = {}
            current_dynamics['gold_label_probs'] = [float(value) for value in cur_gold_label_probs.tolist()]  # Convert each value to float
            current_dynamics['confidence'] = self.confidence(cur_gold_label_probs)
            current_dynamics['variability'] = self.variability(cur_gold_label_probs)
            current_dynamics['correctness'] = self.correctness(cur_gold_label_probs)
            current_dynamics['forgetfulness'] = self.forgetfulness(cur_gold_label_probs)
            self.training_dynamics[idx] = current_dynamics
            
        # Save to JSON
        json_file_name = f'{self.run_name}_training_dynamics.json'
        with open(json_file_name, 'w') as file:
            json.dump(self.training_dynamics, file, ensure_ascii=False, indent=4)

        # Upload to WandB
        wandb.save(json_file_name)

    @staticmethod
    def confidence(gold_label_probs):
        return np.mean(gold_label_probs)

    @staticmethod
    def variability(gold_label_probs):
        return np.std(gold_label_probs)

    @staticmethod
    def correctness(gold_label_probs):
        # TODO
        # UPDATE TO FOLLOW NUMBER OF CLASSES!!!! Not 0.5!!! it's not binary classification
        return np.mean(gold_label_probs > 0.5)

    @staticmethod
    def forgetfulness(gold_label_probs):
        epochs = np.arange(len(gold_label_probs))
        if len(gold_label_probs) == 1:
            return 0
        m, _ = np.polyfit(epochs, gold_label_probs, 1)
        return m