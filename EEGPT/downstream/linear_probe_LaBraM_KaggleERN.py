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

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
import Modules.LaBraM.modeling_finetune

import timm.models
from timm.models import create_model
import torch
from utils import temporal_interpolation
from sklearn import metrics
from utils_eval import get_metrics

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self):
        super().__init__() 
        
        checkpoint = torch.load("Modules/LaBraM/labram-base.pth")
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=2,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        model.load_state_dict(new_checkpoint, strict=False)
        for blk in model.blocks:
            for p in blk.parameters():
                p.requires_grad = False
        self.feature        = model
        self.head   =   LinearWithConstraint(7800, 2, max_norm=1)#
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
    
    def forward(self, x):
        B, C, T = x.shape
        x = x/10
        x = x - x.mean(dim=-2, keepdim=True)
        x = temporal_interpolation(x, 2*200)
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200
        x = x.reshape((B,C,T//200,200))
        pred = self.feature.forward_features(x, input_chans=[i for i in range(C+1)], return_all_tokens=True)
        pred = self.head(pred.flatten(1))
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit.clone().detach().cpu()
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # rocauc = metrics.roc_auc_score(label.clone().detach().cpu(), y_score)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        # self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        
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
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
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
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        # self.log('valid_rocauc', rocauc, on_epoch=True, on_step=False)
        
        return loss
    
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
    def on_train_epoch_end(self) -> None:
            
        label, y_score = [], []
        for x,y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()
    def on_test_epoch_start(self) -> None:
        self.running_scores["test"]=[]
        return super().on_test_epoch_start()
    def on_test_epoch_end(self) -> None:
            
        label, y_score = [], []
        for x,y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('test_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_test_epoch_end()
    
    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.head.parameters())+
            list(self.feature.parameters()),
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
from utils import *
import math
seed_torch(9)
path = "../datasets/downstream"

global max_epochs
global steps_per_epoch
global max_lr

batch_size=64
max_epochs = 100

Folds = {
        1:([12,13,14,16,17,18,20,21,22,23,24,26],[1,3,4,5,8,9,10,15,19,25]),
        2:([2,6,7,11,17,18,20,21,22,23,24,26],[1,3,4,5,8,9,10,15,19,25]),
        3:([2,6,7,11,12,13,14,16,22,23,24,26],[1,3,4,5,8,9,10,15,19,25]),
        4:([2,6,7,11,12,13,14,16,17,18,20,21],[1,3,4,5,8,9,10,15,19,25]),
    } 

for k,v in Folds.items():
    # init model
    model = LitEEGPTCausal()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    training   = read_kaggle_ern_train(path,subjects=v[0])
    validation = read_kaggle_ern_test(path,subjects=v[1])
    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, num_workers=0, shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 4e-4
    trainer = pl.Trainer(accelerator='cuda',
                max_epochs=max_epochs, 
                callbacks=callbacks,
                enable_checkpointing=False,
                logger=[pl_loggers.TensorBoardLogger('./logs/', name="LaBraM_ERN_tb", version=f"fold{k}"), 
                        pl_loggers.CSVLogger('./logs/', name="LaBraM_ERN_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')