import torch
import config
from tqdm import tqdm
from model import LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from datetime import date, time, datetime, timedelta
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
        T.append(datetime.strptime(t, custom_format))

    for i in range(10):
        for j in range(24):
            for k in range(4):
                t = T[-1] + timedelta(minutes=15)
                T.append(t)
                x = torch.DoubleTensor(val).reshape(1, -1)
                pred_y = model(x)
    
    T = T[:-(10*24*4)]


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
