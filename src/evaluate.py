import torch
import config
from torch.utils.data import DataLoader
from dataset import LoadDataset, load_data

def eval(dataloader, model, model_name):
    if model_name != 'lstm':
        model.eval()
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # print('evaluating...')
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


if __name__ == '__main__':
    # device
    torch.cuda.set_device(0)

    # dataset
    dataset_dir = 'dataset/附件1-区域15分钟负荷数据.csv'
    data = load_data(dataset_dir)

    train_data = data[0: int(len(data) * config.TRANING_DATASET_RATIO)]
    dev_data = data[int(len(data) * config.TRANING_DATASET_RATIO): len(data)]

    train_dataset = LoadDataset(train_data)
    dev_dataset = LoadDataset(dev_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)
