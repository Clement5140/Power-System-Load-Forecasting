import os
import torch
import matplotlib.pyplot as plt

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def savefig(save_dir):
    if not os.path.isdir('pic/'):
        os.makedirs('pic/')
    plt.savefig(os.path.join('pic/', save_dir))
