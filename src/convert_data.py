import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from tqdm import tqdm
import os


# libraries = ['fake-circuit-data', 'real-circuit-data']
libraries = ['fake-circuit-data']
benchmark_root = 'benchmark'
data_types = ['_current', '_eff_dist', '_ir_drop_map', '_pdn_density']
dataset_path = 'dataset'

def read_convert_data_raw(data_raw_path, data_types, dataset_path):
    train_data_path = os.path.join(dataset_path, 'train/')
    train_data_csv = os.path.join(dataset_path, 'train_data.csv')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    f = open(train_data_csv, 'w+')
    f.write("training_dataset\n")
    file_list = sorted(os.listdir(data_raw_path))
    for i in tqdm(range(0, len(file_list), len(data_types) + 1)):
        file_prefix = file_list[i].split('.')[0]
        index = int(file_list[i].lstrip("current_map").split('.')[0])
        data_single = {}
        for data_type in data_types:
            filename = file_prefix + data_type + ".csv"
            print(filename)
            df = pd.read_csv(data_raw_path + '/' + filename)
            data_raw = torch.tensor(df.values)
            data_single[data_type[1:]] = data_raw.cpu().numpy()

        data_single_name = "%s_%d.npy"%('current_map', index)
        data_single_file = os.path.join(train_data_path, data_single_name)
        print(data_single_file)
        np.save(data_single_file, data_single)
        f.write(data_single_name + '\n')
    f.close()

for lib in libraries:
    data_raw_path = os.path.join(benchmark_root, lib)
    read_convert_data_raw(data_raw_path, data_types, dataset_path)


