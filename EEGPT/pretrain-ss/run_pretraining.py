"""
Pretrain EEGPT on Khula dataset

Usage:
    python pretrain_eegpt.py --config configs.py

Notes:
- Uses TensorBoard and CSV loggers; add W&B if `sweep=True`
"""
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pytorch_lightning.loggers as pl_loggers
from engine_pretraining import *
import configs
from pytorch_lightning.callbacks import ModelCheckpoint


torch.set_float32_matmul_precision("medium")
seed_torch(0)


def build_loggers(tag, variant, run_name):
    """
    Create the list of loggers used by the Trainer
    """
    folder_dir = f"/scratch/chntzi001/khula/pretrain/ss/10s/logs/23-08/EEGPT_{tag}_{variant}/{run_name}"

    tb_logger = TensorBoardLogger(
        save_dir=str(folder_dir), name=f"{run_name}_tb")
    csv_logger = CSVLogger(save_dir=str(folder_dir), name=f"{run_name}_csv")

    return [tb_logger, csv_logger]


def main():
    # ---------------- Config hyperparameters ---------------- #
    sweep = configs.sweep
    tag = configs.tag
    variant = configs.variant
    timepoints = configs.timepoints
    devices = configs.devices
    max_epochs = configs.max_epochs
    max_lr = configs.max_lr
    batch_size = configs.batch_size

    # ---------------- Create model ---------------- #
    model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), timepoints=timepoints, sweep=sweep, max_epochs=max_epochs, max_lr=max_lr,
                     USE_LOSS_A=(variant != "A"),
                     USE_LN=(variant != "B"),
                     USE_SKIP=(variant != "C"))

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"/scratch/chntzi001/khula/checkpoints/pretrain_khula_eegpt/ss/10s/23-08/EEGPT_{tag}_{variant}/{run_name}",
        filename="EEGPT-{epoch:02d}-{valid_totalLoss:.6f}",
        save_top_k=1,
        monitor="valid_totalLoss",
        mode="min",
        save_last=True
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    logger = build_loggers(tag, variant, run_name)
    if sweep:
        logger.append(wandb_logger)

    trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=[checkpoint_cb, *callbacks],
                         logger=logger)
    print("================== Beginning pretraining ==================")
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
