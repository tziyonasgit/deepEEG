# Training in 256Hz data and 4s
import os
import math
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import copy
import torchvision
from pytorch_lightning import loggers as pl_loggers


from unified.utils import WarmupCosineSchedule, CosineWDSchedule, grad_logger
from modeling_pretraining import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from configs import *
# -- use channels for model

# use_channels_names = ['FP1', 'FPZ', 'FP2',
#                       'AF3', 'AF4',
#                       'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#                       'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
#                       'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
#                       'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#                       'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
#                       'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
#                       'O1', 'OZ', 'O2', ]

use_channels_names = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10',
                      'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22',
                      'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33',
                      'E34', 'E35', 'E36', 'E37', 'E39', 'E40', 'E41', 'E42', 'E45',
                      'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57',
                      'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66',
                      'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
                      'E78', 'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87',
                      'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98',
                      'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108',
                      'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118',
                      'E122', 'E123', 'E124']


class LitEEGPT(pl.LightningModule):

    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True):
        super().__init__()
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN = USE_LN
        self.USE_SKIP = USE_SKIP

        encoder = EEGTransformer(
            img_size=[99, 1000*2],
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])

        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])

        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])

        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.reconstructor = reconstructor
        self.chans_id = encoder.prepare_chan_ids(use_channels_names)

        self.loss_fn = torch.nn.MSELoss()

    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):

        C, N = num_patchs

        while True:
            mask_x = []  # mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random() > p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x) == 0:
                continue
            if len(mask_y_bx) == 0:
                continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape) < p_c_y]
            if len(mask_y_bx) == 0:
                continue
            break

        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0)

    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            C, N = self.encoder.num_patches
            print("x.shape[-1]", x.shape[-1])  # 2000
            print("x.shape[-2]",  x.shape[-2])  # 99
            print("N: ", N)  # 31 -> number of patches -> 2000/99 = 31
            print("C: ", C)  # 99
            assert x.shape[-1] % N == 0 and x.shape[-2] % C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            x = x.permute(0, 3, 1, 2, 4).contiguous()  # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss = loss1 + loss2
        else:
            loss = loss2

        # -- Contrast
        self.log('valid_loss1', loss1, on_epoch=True,
                 on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('valid_loss2', loss2, on_epoch=True,
                 on_step=False, sync_dist=True)
        self.log('valid_loss', loss, on_epoch=True,
                 on_step=False, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss = loss1 + loss2
        else:
            loss = loss2

        # -- Contrast
        self.log('train_loss1', loss1, on_epoch=True,
                 on_step=False, sync_dist=True)
        # -- Reconstruct
        self.log('train_loss2', loss2, on_epoch=True,
                 on_step=False, sync_dist=True)
        self.log('train_loss', loss, on_epoch=True,
                 on_step=False, sync_dist=True)

        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max,
                 on_epoch=True, on_step=False, sync_dist=True)

        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res

    def configure_optimizers(self):

        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs,
                                                           div_factor=2,
                                                           final_div_factor=8,
                                                           pct_start=0.2,
                                                           )
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)
        lr_dict = {
            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1,  # The frequency of the scheduler
            'monitor': 'valid_loss',  # Metric for `ReduceLROnPlateau` to monitor
            'strict': True,  # Whether to crash the training if `monitor` is not found
            'name': None,  # Custom name for `LearningRateMonitor` to use
        }
        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=1e-6,
            final_wd=1e-6,
            T_max=int(max_epochs*steps_per_epoch))
        ema = [0.996, 1.0]
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
                                   for i in range(int(steps_per_epoch*max_epochs)+1))
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


# -- modeling
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
