import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from Modules.models.dn3_ext import ConvEncoderBENDR
from utils_eval import get_metrics

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self):
        super().__init__()    
        self.num_class = 5
        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load("Modules/models/encoder.pt")
        
        self.model = encoder
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        
        self.downsample     = torch.nn.Conv1d(2, 19, 3, stride=3)
        
        self.linear_probe   = torch.nn.Linear(5632, 5)
        
        self.drop           = torch.nn.Dropout(p=0.10)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    
    def forward(self, x):
        x = self.downsample(x)
        
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        
        h = self.model(x)
        
        h = h.flatten(1)
        h = self.drop(h)
        
        pred = self.linear_probe(h)
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        
        return super().on_validation_epoch_end()
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list([self.scale_param])+
            list(self.downsample.parameters())+
            list(self.model.parameters())+
            list(self.linear_probe.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 
####### -- Load Dataset
import torchvision
# Train Data Num : 5Class: 36914 13604 34722 7922 15398
subjects = [0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 75, 76, 77, 81, 82]
N = len(subjects)//10
set_all = set(subjects)
for fold in range(10):
    set_valid = set(subjects[fold*N:(fold+1)*N])
    set_train = set_all - set_valid
    
    train_dataset = torchvision.datasets.DatasetFolder(root="../datasets/downstream/sleep_edf/TrainFold", loader=lambda x: torch.load(x),  extensions=[f'.s{i}' for i in set_train])
    valid_dataset = torchvision.datasets.DatasetFolder(root="../datasets/downstream/sleep_edf/TrainFold", loader=lambda x: torch.load(x), extensions=[f'.s{i}' for i in set_valid])

    # -- begin Training ------------------------------

    import math
    torch.set_float32_matmul_precision('medium' )

    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=8*256

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 1e-5

    # init model
    model = LitEEGPTCausal()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    trainer = pl.Trainer(accelerator='cuda',
                        precision=16,
                        max_epochs=max_epochs, 
                        callbacks=callbacks,
                        logger=[pl_loggers.TensorBoardLogger('./logs/', name="BENDR_SLEEPEDF_tb", version=f"fold{fold+1}"), 
                                pl_loggers.CSVLogger('./logs/', name="BENDR_SLEEPEDF_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')