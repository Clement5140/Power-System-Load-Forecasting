import torch
import config
from tqdm import tqdm
from model import LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from dataset import LoadDataset, load_data2

def eval(dataloader, model, model_name):
    MSE = 0
    for data in dataloader:
        _, x, y = data
        x = x.cuda()
        y = y.cuda()

        pred_y = model(x)
        for i in range(len(y)):
            MSE = MSE + (y.data[i] - pred_y.data[i]) * (y.data[i] - pred_y.data[i])
    MSE = MSE / len(dataloader.dataset)
    return MSE.sum()

MAX = 0
MIN = 0

def eval_(dataloader, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    print('evaluating...')
    MSE = 0
    MAE = 0
    MAPE = 0
    T = []
    Y1 = []
    PRED_Y1 = []
    Y2 = []
    PRED_Y2 = []
    for data in tqdm(dataloader):
        t, x, y = data
        x = x.cuda()
        y = y.cuda()

        pred_y = model(x)
        for i in range(len(y)):
            MSE = MSE + (y.data[i] - pred_y.data[i]) * (y.data[i] - pred_y.data[i])
            MAE = MAE + abs(y.data[i] - pred_y.data[i])
            if y.data[i][0] != 0 and y.data[i][1] != 0:
                MAPE = MAPE + abs(y.data[i] - pred_y.data[i]) / y.data[i]
            T.append(t[i])
            Y1.append(y.data[i][0].item() * (MAX - MIN) + MIN)
            PRED_Y1.append(pred_y.data[i][0].item() * (MAX - MIN) + MIN)
            Y2.append(y.data[i][1].item() * (MAX - MIN) + MIN)
            PRED_Y2.append(pred_y.data[i][1].item() * (MAX - MIN) + MIN)
    MSE = MSE / len(dataloader.dataset)
    MAE = MAE / len(dataloader.dataset)
    MAPE = MAPE / len(dataloader.dataset)

    plt.cla()
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('日期时间')
    plt.ylabel('有功功率最大值/kw')
    plt.plot(T, Y1, label='真实值')
    plt.plot(T, PRED_Y1, label='预测值')
    x_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.show()
    
    plt.cla()
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('日期时间')
    plt.ylabel('有功功率最小值/kw')
    plt.plot(T, Y2, label='真实值')
    plt.plot(T, PRED_Y2, label='预测值')
    x_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.show()
    return MSE, MAE, MAPE

dirs = ['snap_shot/lstm_2_1_best_steps_5000.pt', 'snap_shot/lstm_2_2_best_steps_3800.pt', 'snap_shot/lstm_2_3_best_steps_4000.pt', 'snap_shot/lstm_2_4_best_steps_3800.pt']

if __name__ == '__main__':
    # device
    torch.cuda.set_device(0)

    # dataset
    dataset_dir = 'dataset/附件2-行业日负荷数据.csv'
    data, MAX, MIN = load_data2(dataset_dir)
    data = data[config.TARGET]

    # train_data = data[0: int(len(data) * config.TRANING_DATASET_RATIO)]
    dev_data = data[int(len(data) * config.TRANING_DATASET_RATIO): len(data)]

    # train_dataset = LoadDataset(train_data)
    dev_dataset = LoadDataset(dev_data)

    # train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)

    print('loading model...')
    lstm = LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE, dropout=config.DROPOUT).cuda()
    lstm.load_state_dict(torch.load(dirs[config.TARGET]))

    mse, mae, mape = eval_(dev_dataloader, lstm, 'lstm_2_' + str(config.TARGET+1))
    print('mse: ' + str(mse[0].item()*(MAX-MIN)*(MAX-MIN)) + ' ' + str(mse[1].item()*(MAX-MIN)*(MAX-MIN)))
    print('mae: ' + str(mae[0].item()*(MAX-MIN)) + ' ' + str(mae[1].item()*(MAX-MIN)))
    print('mape: ' + str(mape[0].item()) + ' ' + str(mape[1].item()))
