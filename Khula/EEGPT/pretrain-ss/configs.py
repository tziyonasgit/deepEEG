
import torch
import torchvision
import math
import random
import wandb
from pytorch_lightning.loggers import WandbLogger
import glob
import os
import pickle
from torch.utils.data import Dataset

##### define parameters #####
sweep = True
timepoints = 1024  # for 4s
tag = "tiny1"
variant = "D"
devices = [0]

if sweep:
    global run_name
    wandb.init(
        project="deepEEG"
    )
    config = wandb.config
    max_epochs = 50
    max_lr = config.lr
    batch_size = config.batch_size
    run_name = f"lr{config.lr:.5f}_bs{config.batch_size}"
    wandb.run.name = run_name
    wandb_logger = WandbLogger(
        project="deepEEG", experiment=wandb.run,
        dir="/scratch/chntzi001/wandb_logs"
    )
else:
    max_epochs = 50
    max_lr = 4e-5
    batch_size = 32
    run_name = f"lr{max_lr:.5f}_bs{batch_size}"


class KHULALoader(torch.utils.data.Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(
            open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        print("the shape of X is: ", X.shape)
        Y = -1  # dummy label
        X = torch.FloatTensor(X)
        return X, Y


train_dataset = KHULALoader(
    root="/scratch/chntzi001/khula/processed/train", files=glob.glob("/scratch/chntzi001/khula/processed/train/*.pkl"))
valid_dataset = KHULALoader(
    root="/scratch/chntzi001/khula/processed/val", files=glob.glob("/scratch/chntzi001/khula/processed/val/*.pkl"))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)


steps_per_epoch = math.ceil(len(train_loader)/len(devices))

MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim": 64, "embed_num": 1, "depth": [2, 2, 4], "num_heads": 4},
    "tiny2": {
        "embed_dim": 64, "embed_num": 4, "depth": [2, 2, 4], "num_heads": 4},
    "tiny3": {
        "embed_dim": 64, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "little": {
        "embed_dim": 128, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "base1": {
        "embed_dim": 256, "embed_num": 1, "depth": [6, 6, 6], "num_heads": 4},
    "base2": {
        "embed_dim": 256, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "base3": {
        "embed_dim": 512, "embed_num": 1, "depth": [6, 6, 6], "num_heads": 8},
    "large": {
        "embed_dim": 512, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 8},
}


def get_config(embed_dim=512, embed_num=4, depth=[8, 8, 8], num_heads=4):

    models_configs = {
        'encoder': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'depth': depth[0],
            'num_heads': num_heads,
        },
        'predictor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'predictor_embed_dim': embed_dim,
            'depth': depth[1],
            'num_heads': num_heads,
        },
        'reconstructor': {
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'reconstructor_embed_dim': embed_dim,
            'depth': depth[2],
            'num_heads': num_heads,
        },
    }
    return models_configs
