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

    # import matplotlib.pyplot as plt
    # from matplotlib.pyplot import MultipleLocator
    # plt.cla()
    # plt.clf()
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.xlabel('日期时间')
    # plt.ylabel('总有功功率/kw')
    # plt.plot(time, v)
    # x_major_locator = MultipleLocator(100000)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xticks(rotation=30)
    # plt.show()

    for i in tqdm(range(len(v)-96)):
        train_seq = []
        train_label = []
        for j in range(i, i+96):
            train_seq.append([v[j]])
        train_label.append(v[i + 96])
        train_seq = torch.DoubleTensor(train_seq)
        train_label = torch.DoubleTensor(train_label).view(-1)
        data.append((time[i+96], train_seq, train_label))

    return data, MAX, MIN


id = {'大工业用电': 0, '非普工业': 1, '普通工业': 2, '商业': 3}

def load_data2(dir=None):
    data = [[], [], [], []]
    time = [[], [], [], []]
    v1 = [[], [], [], []]
    v2 = [[], [], [], []]
    MAX = 0.0
    MIN = 10000000.0

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
            time[id[row[0]]].append(row[1])
            a = float(row[2])
            b = float(row[3])
            v1[id[row[0]]].append(a)
            v2[id[row[0]]].append(b)
            if a > MAX:
                MAX = a
            if b > MAX:
                MAX = b
            if a < MIN:
                MIN = a
            if b < MIN:
                MIN = b

        for i in range(4):
            # import matplotlib.pyplot as plt
            # from matplotlib.pyplot import MultipleLocator
            # plt.cla()
            # plt.clf()
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False
            # plt.xlabel('日期时间')
            # plt.ylabel('总有功功率/kw')
            # plt.plot(time[i], v1[i])
            # x_major_locator = MultipleLocator(100)
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(x_major_locator)
            # plt.xticks(rotation=30)
            # plt.show()
            for j in range(len(v1[i])):
                v1[i][j] = (v1[i][j] - MIN) / (MAX - MIN)
                v2[i][j] = (v2[i][j] - MIN) / (MAX - MIN)

    for t in range(4):
        for i in tqdm(range(len(v1[t])-30)):
            train_seq = []
            train_label = []
            for j in range(i, i+30):
                train_seq.append([v1[t][j], v2[t][j]])
            train_label.append([v1[t][i+30], v2[t][i+30]])
            train_seq = torch.DoubleTensor(train_seq)
            train_label = torch.DoubleTensor(train_label).view(-1)
            data[t].append((time[t][i+30], train_seq, train_label))

    return data, MAX, MIN
