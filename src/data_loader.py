import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os


class DPDataset(Dataset):
    def __init__(self, csv_file, root_dir, device=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.data_types = ['current', 'eff_dist', 'ir_drop_map', 'pdn_density']

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx, 0])
        data = np.load(data_name,allow_pickle=True).item()
        return data['current'], data['eff_dist'], data['ir_drop_map'], data['pdn_density']
