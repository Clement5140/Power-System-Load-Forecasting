import torch
import utils
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import LoadDataset, load_data
from model import LSTM
from evaluate import eval

# device
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
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

# model
lstm = LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=config.OUTPUT_SIZE, dropout=config.DROPOUT).cuda()
lstm_criterion = torch.nn.MSELoss().cuda()
lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

def train(model, crit, opt, model_name):
    print('training ' + model_name + '...')
    model.train()
    cur_step = 0
    best_mse = 10000
    losses = []
    for epoch in range(config.N_EPOCHES):
        print('epoch %d/%d:' % (epoch+1, config.N_EPOCHES))

        if epoch % 4 == 0:
            for p in opt.param_groups:
                p['lr'] *= 0.9

        tk0 = tqdm(train_dataloader)
        for data in tk0:
            _, x, y = data
            x = x.cuda()
            y = y.cuda()

            pred_y = model(x)

            opt.zero_grad()
            loss = crit(pred_y, y)
            
            loss.backward()
            opt.step()

            losses += [loss.item()]

            cur_step += 1
            if cur_step % config.DISPLAY_STEP == 0:
                std = (pred_y - y).std()
                errors = {
                    'loss': loss.item(),
                    'std' : std.item(),
                }
                tk0.set_postfix(errors)

            if cur_step % config.SAVE_STEP == 0:
                dev_mse = eval(dev_dataloader, model, model_name)
                if dev_mse <= best_mse:
                    best_mse = dev_mse
                    utils.save(model, 'snap_shot', model_name+'_best', cur_step)
                print('\rcur_step:%d mse:%.7f' % (cur_step, dev_mse))

                step_bins = 20
                num_examples = (len(losses) // step_bins) * step_bins
                plt.cla()
                plt.clf()
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Loss"
                )
                plt.legend()
                utils.savefig(model_name+'_line_{0}.png'.format(cur_step))

if __name__ == '__main__':
    train(lstm, lstm_criterion, lstm_optimizer, 'lstm')
