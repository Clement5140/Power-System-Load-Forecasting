import torch
import config
from tqdm import tqdm
from model import LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from dataset import LoadDataset, load_data

def eval(dataloader, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    MSE = 0
    for data in dataloader:
        _, x, y = data
        x = x.cuda()
        y = y.cuda()

        pred_y = model(x)
        for i in range(len(y)):
            MSE = MSE + (y.data[i] - pred_y.data[i]) * (y.data[i] - pred_y.data[i])
    MSE = MSE / len(dataloader.dataset)
    return MSE

MAX = 0
MIN = 0

def eval_(dataloader, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    print('evaluating...')
    MSE = 0
    T = []
    Y = []
    PRED_Y = []
    for data in tqdm(dataloader):
        t, x, y = data
        x = x.cuda()
        y = y.cuda()

        pred_y = model(x)
        for i in range(len(y)):
            MSE = MSE + (y.data[i] - pred_y.data[i]) * (y.data[i] - pred_y.data[i])
            T.append(t[i])
            Y.append(y.data[i].item() * (MAX - MIN) + MIN)
            PRED_Y.append(pred_y.data[i].item() * (MAX - MIN) + MIN)
    MSE = MSE / len(dataloader.dataset)

    plt.cla()
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('日期时间')
    plt.ylabel('总有功功率/kw')
    plt.plot(T, Y, label='真实值')
    plt.plot(T, PRED_Y, label='预测值')
    x_major_locator = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.show()
    return MSE


if __name__ == '__main__':
    # device
    torch.cuda.set_device(0)

    # dataset
    dataset_dir = 'dataset/附件1-区域15分钟负荷数据.csv'
    data, MAX, MIN = load_data(dataset_dir)

    # train_data = data[0: int(len(data) * config.TRANING_DATASET_RATIO)]
    dev_data = data[int(len(data) * config.TRANING_DATASET_RATIO): len(data)]

    # train_dataset = LoadDataset(train_data)
    dev_dataset = LoadDataset(dev_data)

    # train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)

    print('loading model...')
    lstm = LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE, dropout=config.DROPOUT).cuda()
    lstm.load_state_dict(torch.load('snap_shot/lstm_best_steps_26000.pt'))

    mse = eval_(dev_dataloader, lstm, 'lstm')
    print(mse.item())
