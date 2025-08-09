import math
from utils import *
from utils_eval import get_metrics
from sklearn import metrics
from utils import temporal_interpolation
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from Modules.models.EEGPT_mcae import EEGTransformer
import random
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os
import tqdm
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F
import pickle
from scipy.signal import resample


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(7)

use_channels_names = ['PZ', 'C2', 'P5', 'P6', 'TP8', 'C5', 'FC4', 'FT7', 'AF4', 'POZ', 'F6', 'TP7', 'PO7', 'PO4', 'O2', 'F8', 'F4', 'T7', 'CP6', 'PO8', 'C3', 'CP1', 'CP4', 'F3', 'OZ', 'FC3',
                      'FT8', 'F7', 'FP2', 'PO3', 'P4', 'F5', 'FC2', 'P2', 'AF3', 'CPZ', 'F2', 'CP5', 'FP1', 'FC1', 'P1', 'FZ', 'FPZ', 'CP3', 'O1', 'P3', 'C6', 'FC6', 'C4', 'F1', 'CP2', 'FCZ', 'FC5', 'C1']

early_stopping = EarlyStopping(
    monitor='valid_accuracy',  # metric to monitor
    patience=10,          # epochs with no improvement after which training will stop
    mode='max',          # mode for min loss; 'max' if maximizing metric
    min_delta=0.001      # minimum change to qualify as an improvement
)


def prepare_KHULA_dataset(filepath):

    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(filepath, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(filepath, "val"))
    test_files = os.listdir(os.path.join(filepath, "test"))

    print(len(train_files), len(val_files), len(test_files))

    train_dataset = KHULALoader(os.path.join(filepath, "train"), train_files)
    test_dataset = KHULALoader(os.path.join(filepath, "test"), test_files)
    val_dataset = KHULALoader(os.path.join(filepath, "val"), val_files)

    return train_dataset, test_dataset, val_dataset


class KHULALoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.labels = {"3": 1, "6": 2, "12": 3, "24": 4}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(
            open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)

        class_label = sample["y"]

        if class_label not in self.labels:
            raise ValueError(f"Unexpected label: {class_label}")

        Y = self.labels[class_label] - 1
        X = torch.FloatTensor(X)
        return X, Y


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="/home/chntzi001/deepEEG/EEGPT/downstream/Checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()
        self.chans_num = len(use_channels_names)
        target_encoder = EEGTransformer(
            img_size=[54, 2*256],
            patch_size=32*2,
            patch_stride=32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)
        print("The self.chans_id is......",  self.chans_id)

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path, weights_only=False)

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)
        # Freeze all encoder parameters -> test below and see what
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.chan_conv = Conv1dWithConstraint(
            54, self.chans_num, 1, max_norm=1)

        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(240, 4, max_norm=0.25)

        self.drop = torch.nn.Dropout(p=0.50)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):
        B, C, T = x.shape
        x = x/10
        x = x - x.mean(dim=-2, keepdim=True)
        x = temporal_interpolation(x, 2*256)
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)

        h = self.linear_probe1(self.drop(h))

        h = h.flatten(1)

        h = self.linear_probe2(h)

        return x, h

    def training_step(self, batch, batch_idx):
        print("IN THE TRAINING STEP METHOD")
        # gets called for every batch
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        # print("logits shape: ", logit.shape)
        # [64, 4] -> [batch size, num classes]
        # print("logits are: ", logit)
        preds = torch.argmax(logit, dim=-1)
        # print("preds shape: ", preds.shape)
        # [64] -> [batch size]
        # print("preds are: ", preds)
        accuracy = ((preds == label)*1.0).mean()
        y_score = logit.clone().detach().cpu()
        y_score = torch.softmax(y_score, dim=-1)
        print("in training step, y_score is: ", y_score)
        self.running_scores["train"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # rocauc = metrics.roc_auc_score(label.clone().detach().cpu(), y_score)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)

        label = torch.cat(label, dim=0)  # consolidates into one tensor
        y_score = torch.cat(y_score, dim=0)

        if is_binary:
            metrics = ["accuracy", "balanced_accuracy", "precision",
                       "recall", "cohen_kappa", "f1", "roc_auc"]
        else:
            metrics = ["accuracy", "balanced_accuracy",
                       "cohen_kappa", "f1_weighted"]

        print("y_score is-------> ", y_score)
        print("y_score shape is ", y_score.shape)
        # y_score is output in get_metrics
        # and output is y_prob in pyheath multiclass_metrics_fn
        # y_prob needs to be [n_samples, n_classes] and an nparray
        # need predicted probabilities so softmax function

        results = get_metrics(y_score.cpu().numpy(),
                              label.cpu().numpy(), metrics, is_binary)

        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True,
                     on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        print("IN THE VALIDATION STEP METHOD")
        # gets called for every batch
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        # Should be [batch_size, num_classes]
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label)*1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)
        self.running_scores["valid"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:

        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        # rocauc = metrics.roc_auc_score(label, y_score)
        # self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:

        label, y_score = [], []
        for x, y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        # rocauc = metrics.roc_auc_score(label, y_score)
        # self.log('test_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label)*1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["test"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1,  # The frequency of the scheduler
            'monitor': 'val_loss',  # Metric for `ReduceLROnPlateau` to monitor
            'strict': True,  # Whether to crash the training if `monitor` is not found
            'name': None,  # Custom name for `LearningRateMonitor` to use
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


# load configs
global max_epochs
global steps_per_epoch
global max_lr
global is_binary
seed_torch(9)
batch_size = 64
max_epochs = 100
numClasses = 4
filepath = "/scratch/chntzi001/khula/processed/"
is_binary = False

train_dataset, test_dataset, val_dataset = prepare_KHULA_dataset(
    filepath)

# init model
model = LitEEGPTCausal()
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor, early_stopping]

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

steps_per_epoch = math.ceil(len(train_loader))
max_lr = 5e-5

accelerator = "cuda" if torch.cuda.is_available() else "cpu"

print("max learning rate is: ", max_lr)
trainer = pl.Trainer(accelerator,
                     max_epochs=max_epochs,
                     callbacks=callbacks,
                     enable_checkpointing=False,
                     logger=[pl_loggers.TensorBoardLogger('/home/chntzi001/deepEEG/EEGPT/downstream/linearprobe/log/', name="EEGPT_KHULA_tb"),
                             pl_loggers.CSVLogger('/home/chntzi001/deepEEG/EEGPT/downstream/linearprobe/log/', name="EEGPT_KHULA_csv")])

trainer.fit(model, train_loader, valid_loader)
