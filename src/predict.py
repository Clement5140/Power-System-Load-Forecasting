import csv
import torch
import config
from model import LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from datetime import datetime, timedelta
from dataset import load_data

MAX = 0
MIN = 0
custom_format = '%Y/%m/%d %H:%M:%S'

def predict(data, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    print('predicting...')

    T = []
    val = []
    for i in range(95, -1, -1):
        t, _, y = data[-i-1]
        val.append(y.item())
        T.append(t)

    Y = []
    days = 10
    last_t = datetime.strptime(T[-1], custom_format)
    for _ in range(days):
        for _ in range(24):
            for _ in range(4):
                t = last_t + timedelta(minutes=15)
                last_t = t
                T.append(t.strftime(custom_format))
                x = torch.DoubleTensor(val[-96:]).cuda().reshape(1, -1, 1)
                y = model(x)
                val.append(y.data[0].item())
                Y.append(y.data[0].item() * (MAX - MIN) + MIN)

    T = T[-(days*24*4):]

    plt.cla()
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('日期时间')
    plt.ylabel('总有功功率/kw')
    plt.plot(T, Y)
    x_major_locator = MultipleLocator(96)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.show()

    with open('result.csv', 'w', encoding='utf-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(('数据时间', '总有功功率（kw）'))
        for i in range(len(T)):
            f_csv.writerow((T[i], Y[i]))

    T = []
    val = []
    for i in range(95, -1, -1):
        t, _, y = data[-i-1]
        val.append(y.item())
        T.append(t)

    Y = []
    days = 92
    MAXX = 0.0
    MINN = 100000000.0
    MAX_T = None
    MIN_T = None
    last_t = datetime.strptime(T[-1], custom_format)
    for _ in range(days):
        for _ in range(24):
            for _ in range(4):
                t = last_t + timedelta(minutes=15)
                last_t = t
                x = torch.DoubleTensor(val[-96:]).cuda().reshape(1, -1, 1)
                y = model(x)
                val.append(y.data[0].item())
                true_y = y.data[0].item() * (MAX - MIN) + MIN
                Y.append(true_y)
                if true_y > MAXX:
                    MAXX = true_y
                    MAX_T = t
                if true_y < MINN:
                    MINN = true_y
                    MIN_T = t
    print('max: ' + str(MAXX) + ' time: ' + MAX_T.strftime(custom_format))
    print('min: ' + str(MINN) + ' time: ' + MIN_T.strftime(custom_format))


if __name__ == '__main__':
    # device
    torch.cuda.set_device(0)

    # data
    dataset_dir = 'dataset/附件1-区域15分钟负荷数据.csv'
    data, MAX, MIN = load_data(dataset_dir)

    print('loading model...')
    lstm = LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE, dropout=config.DROPOUT).cuda()
    lstm.load_state_dict(torch.load('snap_shot/lstm_best_steps_26000.pt'))

    predict(data, lstm, 'lstm')
