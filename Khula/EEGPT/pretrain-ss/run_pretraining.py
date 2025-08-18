
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pytorch_lightning.loggers as pl_loggers
from engine_pretraining import *
import configs
from pytorch_lightning.callbacks import ModelCheckpoint


torch.set_float32_matmul_precision("medium")
seed_torch(0)

####### hyperparams from configs ######
sweep = configs.sweep
tag = configs.tag
variant = configs.variant
timepoints = configs.timepoints
devices = configs.devices
max_epochs = configs.max_epochs
max_lr = configs.max_lr
batch_size = configs.batch_size


model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), timepoints=timepoints, sweep=sweep, max_epochs=max_epochs, max_lr=max_lr,
                 USE_LOSS_A=(variant != "A"),
                 USE_LN=(variant != "B"),
                 USE_SKIP=(variant != "C"))


checkpoint_cb = ModelCheckpoint(
    # where to save
    dirpath=f"/scratch/chntzi001/khula/checkpoints/pretrain_khula_eegpt/ss/18-08/EEGPT_{tag}_{variant}/{run_name}",
    filename="EEGPT-{epoch:02d}-{valid_totalLoss:.4f}",
    save_top_k=1,
    monitor="valid_totalLoss",
    mode="min",
    save_last=True
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

if sweep:
    logger = [pl_loggers.TensorBoardLogger(f'/home/chntzi001/deepEEG/EEGPT/pretrain/ss/logs/18-08/EEGPT_{tag}_{variant}/{run_name}', name=f"{run_name}_tb"),
              pl_loggers.CSVLogger(f'/home/chntzi001/deepEEG/EEGPT/pretrain/logs/ss/18-08/"EEGPT_{tag}_{variant}/{run_name}', name=f"{run_name}_csv"), wandb_logger]
else:
    logger = [TensorBoardLogger(f'/home/chntzi001/deepEEG/EEGPT/pretrain/ss/logs/18-08/EEGPT_{tag}_{variant}/{run_name}', name=f"{run_name}_tb"),
              CSVLogger(f'/home/chntzi001/deepEEG/EEGPT/pretrain/logs/ss/18-08/"EEGPT_{tag}_{variant}/{run_name}', name=f"{run_name}_csv")]
trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=[checkpoint_cb, *callbacks],
                     logger=logger)
trainer.fit(model, train_loader, valid_loader)
