import torch
import numpy as np
from torch.utils.data import Dataset


class FlexDataset(Dataset):
    def __init__(self, inputs, targets, transform=None, target_transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_inputs = self.inputs[idx]
        batch_targets = self.targets[idx]

        return batch_inputs, batch_targets