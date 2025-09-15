# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
from torchmetrics import ConfusionMatrix
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange
import csv
import numpy as np
import wandb
import os
theOutputs = {}


def getVarOutputs():
    return theOutputs


def train_class_batch(model, samples, target, criterion, ch_names):
    target = target.float()
    outputs = model(samples)
    target = target.view_as(outputs)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('MAE_batch', utils.SmoothedValue(
        window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * \
                        param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=64)
        numPatches = samples.shape[2]
        targets = targets.to(device, non_blocking=True)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.amp.autocast(device_type='cuda'):
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        torch.cuda.empty_cache()

        loss_value = float(loss.item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            print("loss:", float(loss))
            print("head grad norm:", sum(
                (p.grad.norm().item() if p.grad is not None else 0.0)
                for p in model.head.parameters()
            ))
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(MAE_batch=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(MAE_batch=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(epoch, data_loader, model, device, args, header='Test:', ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    criterion = torch.nn.L1Loss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('MAE_batch', utils.SmoothedValue(
        window_size=1, fmt='{value:.4f}'))

    # switch to evaluation mode
    model.eval()

    pred = []
    true = []

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]  # batch[0] is the EEG data - X
        target = batch[-1]  # batch[-1] is the target/label - y
        target = target.to(device, non_blocking=True).to(torch.float32)
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=64)
        print("EEG shape is: ", EEG.shape)

        # compute output
        with torch.amp.autocast(device_type='cuda'):
            yHat = model(EEG)
            batch_mae = float(criterion(yHat, target).item())
        print("pred stats:", float(yHat.min()),
              float(yHat.max()), float(yHat.mean()))

        if step == 0:
            print("target sample:", target[:16].tolist())
            print("target stats:", float(target.min()),
                  float(target.max()), float(target.mean()))

        # collecting prediction probabilities and true labels
        pred.append(yHat.detach().cpu())
        true.append(target.detach().cpu())

        metric_logger.update(MAE_batch=batch_mae)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    y_pred = torch.cat(pred, dim=0).float()
    print("Number of predictions: ", y_pred.shape[0])
    y_true = torch.cat(true, dim=0).float()

    diff = (y_pred - y_true)
    mae = diff.abs().mean().item()
    rmse = (diff.pow(2).mean().sqrt()).item()
    y_mean = y_true.mean()
    SS_tot = ((y_true - y_mean) ** 2).sum()
    SS_res = ((y_true - y_pred) ** 2).sum()
    r2 = (1.0 - SS_res / (SS_tot + 1e-12)).item()

    if header == "Test:":
        output_csv = os.path.join(args.log_dir, "predictionsTest.csv")
        try:
            with open(output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                yp = y_pred.numpy().tolist()
                yt = y_true.numpy().tolist()
                for p, t in zip(yp, yt):
                    writer.writerow([epoch, p, t])
        except Exception as e:
            print(f"Warning! Could not append predictions CSV: {e}")

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'mae_batch_avg': metric_logger.meters['MAE_batch'].global_avg
    }
