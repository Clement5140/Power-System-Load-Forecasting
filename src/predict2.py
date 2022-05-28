import csv
import torch
import config
from model import LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from datetime import datetime, timedelta
from dataset import load_data2

MAX = 0
MIN = 0
custom_format = '%Y/%m/%d'

dirs = ['snap_shot/lstm_2_1_best_steps_5000.pt', 'snap_shot/lstm_2_2_best_steps_3800.pt', 'snap_shot/lstm_2_3_best_steps_4000.pt', 'snap_shot/lstm_2_4_best_steps_3800.pt']

def predict(data, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    print('predicting...')

    T = []
    val = []
    for i in range(29, -1, -1):
        t, _, y = data[-i-1]
        val.append([y[0].item(), y[1].item()])
        T.append(t)

    Y1 = []
    Y2 = []
    days = 92
    last_t = datetime.strptime(T[-1], custom_format)
    for _ in range(days):
        t = last_t + timedelta(days=1)
        last_t = t
        T.append(t.strftime(custom_format))
        x = torch.DoubleTensor(val[-30:]).cuda().reshape(1, -1, 2)
        y = model(x)
        y = y[0]
        val.append([y[0].item(), y[1].item()])
        Y1.append(y[0].item() * (MAX - MIN) + MIN)
        Y2.append(y[1].item() * (MAX - MIN) + MIN)

    T = T[-days:]

    plt.cla()
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('日期时间')
    plt.ylabel('有功功率最大值/kw')
    plt.plot(T, Y1)
    x_major_locator = MultipleLocator(20)
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
    plt.plot(T, Y2)
    x_major_locator = MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.show()

    with open('result' + str(config.TARGET+1) + '.csv', 'w', encoding='utf-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(('数据时间', '有功功率最大值（kw）', '有功功率最小值（kw）'))
        for i in range(len(T)):
            f_csv.writerow((T[i], Y1[i], Y2[i]))

if __name__ == '__main__':
    # device
    torch.cuda.set_device(0)

    # data
    dataset_dir = 'dataset/附件2-行业日负荷数据.csv'
    data, MAX, MIN = load_data2(dataset_dir)
    data = data[config.TARGET]

    print('loading model...')
    lstm = LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE, dropout=config.DROPOUT).cuda()
    lstm.load_state_dict(torch.load(dirs[config.TARGET]))

    predict(data, lstm, 'lstm_2_' + str(config.TARGET+1))
