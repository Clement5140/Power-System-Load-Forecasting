import csv
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, data=None):
        self.data = data
        self.len = len(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.len


def load_data(dir=None):
    data = []
    time = []
    v = []
    MAX = 0
    MIN = 0

    print('loading data...')
    with open(dir, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter=',')

        rows = []
        first_line = True
        for row in f_csv:
            if first_line:
                first_line = False
                continue
            rows.append(row)

        for row in rows:
            time.append(row[0])
            v.append(float(row[1]))
        
        MAX = np.max(v)
        MIN = np.min(v)
        v = (v - MIN) / (MAX - MIN)

    for i in tqdm(range(len(v)-96)):
        train_seq = []
        train_label = []
        for j in range(i, i+96):
            train_seq.append([v[j]])
        train_label.append(v[i + 96])
        train_seq = torch.DoubleTensor(train_seq)
        train_label = torch.DoubleTensor(train_label).view(-1)
        data.append((time[i], train_seq, train_label))

    return data, MAX, MIN
