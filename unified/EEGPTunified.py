import torch
import numpy as np
import os
import pickle
from scipy.signal import resample
import csv
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
from torch import inf
import math
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
import time
from typing import Iterable, Optional
from collections import defaultdict, deque
from einops import rearrange
import sys
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import torch
from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
import json
import torch.nn as nn
import datetime
from functools import partial
from logging import getLogger
logger = getLogger()
import torch.distributed as dist

standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(
        args)

    output_csv = os.path.join(args.log_dir, "predictionsss.csv")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "True Label"])

    sampler_train = torch.utils.data.SequentialSampler(
        dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = TensorboardLogger(log_dir=args.log_dir)
    
    print("Setting up training dataset")
    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            drop_last=True,
        )
    print("Setting up validation dataset")
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            drop_last=False
        )
    print("Setting up test dataset")
    data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                drop_last=False
            )

    model = get_models(args)
    patch_size = 64  
    window_size = (1, args.input_size // patch_size)
    print("patch_size: ", patch_size)
    print("window_size: ", window_size)
    
    checkpoint = torch.load(
                args.finetune, map_location='cpu', weights_only=False)
    print("================== Loading checkpoint =================")
    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = checkpoint['state_dict']
    load_state_dict(model, checkpoint_model)
    
    model.to(device)
    model_ema = None
    model_without_ddp = model
    
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.learning_rate)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training per epoch = %d" %
          num_training_steps_per_epoch)
    
    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None
        
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    skip_weight_decay_list = model.no_weight_decay()
    
    optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScalerWithGradNormCount()

    print("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))
    
    print("Using CrossEntropyLoss")
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(
                epoch * num_training_steps_per_epoch * args.update_freq)
        print("got here......")
        train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                ch_names=ch_names, is_binary=args.nb_classes == 1
            )
        print("got here......though")
        if data_loader_val is not None:
            print("============== Evaluating on validation and test set ==============")
            print("Here batch size is = %d" % int(1.5 * args.batch_size))
            val_stats = evaluate(data_loader_val, model, device, args, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(
                f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test, model, device, args, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(
                f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                max_accuracy_test = test_stats["accuracy"]
                bestEpoch = epoch
            print(
                f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(
                            accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(
                            balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value,
                                          head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(
                            roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value,
                                          head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                    elif key == 'class_acc':
                        try:
                            log_writer.update(
                                class_acc=value, head="test", step=epoch)
                        except Exception as e:
                            print(f"⚠️val error with logging class_acc")
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(
                            accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(
                            balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value,
                                          head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(
                            pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(
                            roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value,
                                          head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)
                    elif key == 'class_acc':
                        try:
                            log_writer.update(
                                class_acc=value, head="test", step=epoch)
                        except Exception as e:
                            print(f"⚠️test error with logging class_acc")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        
def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs
        
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = get_input_chans(ch_names)
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
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
        targets = targets.to(device, non_blocking=True)
       
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)
            else:
                with torch.amp.autocast(device_type='cpu'):
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
  
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:

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
            
        class_acc = (output.max(-1)[-1] ==
                         targets.squeeze()).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
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

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
     
def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

@torch.no_grad()
def evaluate(data_loader, model, device, args, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    
    model.eval()

    pred = []
    true = []

    output_csv = os.path.join(args.log_dir, "predictionsss.csv")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "True Label"])

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]  # batch[0] is the EEG data - X
        target = batch[-1]  # batch[-1] is the target/label - y
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=64)
        print("EEG shape is: ", EEG.shape)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                output = model(EEG)
                loss = criterion(output, target)
        else:
            with torch.amp.autocast(device_type='cpu'):
                output = model(EEG)
                loss = criterion(output, target)

        print("output shape: ", output.shape)

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
            output = torch.nn.functional.softmax(output, dim=1)
        target = target.cpu()

        results = get_metrics(
            output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        if not is_binary:
            preds_list = torch.argmax(output, dim=1)
            preds_list = preds_list.tolist()
            target_list = target.tolist()
            print("preds_class: ", preds_list)
            print("target_class: ", target_list)

        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(preds_list, target_list))

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    print("pred currently: ", pred)
    predtensor = torch.cat(pred, dim=0)
    print("predtensor: ", predtensor)
    predtensor = torch.argmax(predtensor, dim=1)
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret


    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        if torch.cuda.is_available():
            self._scaler = torch.amp.GradScaler('cuda')
        else:
            self._scaler = torch.amp.GradScaler('cpu')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(
                    f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm
    
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    print("Loading state dictionary...")
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_models(args):
    # full set of EEG channel names available in your dataset -> 54
    use_channels_names = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
                          'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

    # full set of EEG channel names available in your dataset -> 54
    ch_names = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
                'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

    model = EEGPTClassifier(
        num_classes=args.nb_classes,
        in_channels=len(ch_names),
        img_size=[len(use_channels_names), 1024],
        use_channels_names=use_channels_names,
        use_chan_conv=True,
        use_mean_pooling=args.use_mean_pooling, logdir=args.log_dir)
    
    print("made model")

    return model

class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(
                head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def get_dataset(args):
    print("Preparing KHULA dataset...")
    
    if args.hpc:
        datasetroot = f"/scratch/chntzi001/khula/processed"
    else:
        datasetroot = "/Users/cccohen/deepEEG/unified/khuladataset"
    train_dataset, test_dataset, val_dataset = prepare_KHULA_dataset(datasetroot)

    ch_names = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
                'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']

    args.nb_classes = 4
    metrics = ["accuracy", "balanced_accuracy",
               "cohen_kappa", "f1_weighted"]
    
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def prepare_KHULA_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = KHULALoader(os.path.join(root, "train"), train_files)
    test_dataset = KHULALoader(os.path.join(root, "test"), test_files)
    val_dataset = KHULALoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
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

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        # to prevent all 0 or all 1 and raise the AUROC error
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:

        # y_true is target, y_prob is output
        y_true = target
        y_prob = output
        try:
            results = multiclass_metrics_fn(
                y_true=target, y_prob=output, metrics=metrics
            )
        except Exception as e:
            print("Metric computation failed!")
            print("y_true:", y_true)
            print("y_prob:", y_prob)
            raise e

    return results


# CHANNEL_DICT contains the numbered order of channel names in the checkpoint
CHANNEL_DICT = {k.upper(): v for v, k in enumerate(['FP1', 'FPZ', 'FP2',
                                                    'AF3', 'AF4',
                                                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                                                    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                                                    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                                                    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                                                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                                                    'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
                                                    'O1', 'OZ', 'O2', ]
                                                   )}
# current data channel list AND is actually what is passed as use_channel_names
chOrder_standard = ['FPZ', 'POZ', 'P7', 'OZ', 'P8', 'T8', 'AF4', 'CP2', 'PO4', 'CP4', 'FC6', 'C1', 'CP5', 'AF3', 'CP1', 'FZ', 'F1', 'CZ', 'PZ', 'F4', 'P3', 'F8', 'TP7', 'C6', 'O1', 'FC3',
                    'C2', 'TP8', 'FC5', 'FCZ', 'C4', 'F3', 'FP2', 'CP6', 'FC2', 'F7', 'P1', 'PO8', 'FT8', 'CP3', 'T7', 'PO7', 'PO3', 'P4', 'FC4', 'O2', 'C5', 'P6', 'C3', 'P5', 'FT7', 'FC1', 'P2', 'F2']


################################# Utils ######################################


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """
    B, N, C, D = x.shape
    if len(mask.shape) == 2:
        mN, mC = mask.shape

        mask_keep = mask.reshape((1, mN*mC, 1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)),
                                dim=-2, index=mask_keep)
        masked_x = masked_x.contiguous().view((B, mN, mC, D))
    else:
        mN = mask.shape[0]

        mask_keep = mask.reshape((1, mN, 1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)),
                                dim=-2, index=mask_keep)
    return masked_x


def apply_mask_t(mask_t, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """
    B, N, D = x.shape
    mN = mask_t.shape[0]

    mask_keep = mask_t.reshape((1, mN, 1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

# helper functions


def exists(val):
    return val is not None

# rotary embedding helper functions


def rotate_half(x):

    # x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x = x.reshape((*x.shape[:-1], x.shape[-1]//2, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    # return rearrange(x, '... d r -> ... (d r)')
    return x.flatten(-2)


def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[...,
                                                 start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)

################################# RoPE Model Begin ######################################


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta=10000,
        learned_freq=False,
        interpolate_factor=1.
    ):
        super().__init__()

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)
                  [:(dim // 2)].float() / dim)),
            requires_grad=learned_freq)

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        self.register_buffer('scale', None)

    def prepare_freqs(self, num_patches=(1, 8), device='cuda', dtype=torch.float, offset=0):
        # num_patches (C, N)
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        seq_pos = torch.arange(N, device=device, dtype=dtype)
        seq_pos = seq_pos.repeat_interleave(
            repeats=C, dim=0)  # correspond to x (B, N, C, D)
        seq_pos = (seq_pos + offset) / self.interpolate_factor

        freqs = self.freqs
        freqs = torch.outer(seq_pos.type(freqs.dtype),
                            freqs)  # (n_seq_pos, n_freqs)
        freqs = freqs.repeat_interleave(
            repeats=2, dim=-1)    # (n_seq_pos, n_freqs*2)

        self.cache[cache_key] = freqs

        return freqs

################################# EEGPT Model Begin ######################################


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention = return_attention

    def forward(self, x, freqs=None):
      
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C //
                                  # 3,B,nh,t,d
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,nh,t,d

        if self.use_rope:  # RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(
                    q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(
                    torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(
            B, T, C)  # (B, nh, T, hs) -> (B, T, hs*nh)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
        super().__init__()

        self.return_attention = True
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal, use_rope=use_rope, return_attention=return_attention)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        print("y: ", y)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)
        print("x: ", x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = (
                (img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size if patch_stride is None else patch_stride))

    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x).transpose(1, 3)  # B, T, C, D
        return x


################################# Finetune Model Begin ######################################
class EEGTransformerReconstructor(nn.Module):
    """ EEG Transformer """

    def __init__(
        self,
        num_patches,
        patch_size=64,
        embed_num=1,
        use_pos_embed=False,
        use_inp_embed=True,
        embed_dim=768,
        reconstructor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor=2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches

        # --
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, reconstructor_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        # --
        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(
                embed_dim, reconstructor_embed_dim, bias=True)

        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, reconstructor_embed_dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # --
        self.time_embed_dim = (reconstructor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(
            dim=self.time_embed_dim, interpolate_factor=interpolate_factor)

        self.chan_embed = nn.Embedding(
            len(CHANNEL_DICT), reconstructor_embed_dim)
        # --
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True,
                return_attention=(i+1) == return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(
            reconstructor_embed_dim, patch_size, bias=True)
        # ------
        self.init_std = init_std

    def get_num_layers(self):
        return len(self.reconstructor_blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed', 'chan_embed'}

    def forward(self, x):
        # -- map from encoder-dim to pedictor-dim
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N = self.num_patches
        B, mN, eN, D = x.shape

        # assert mN == N, f"{mN},{N}"
        # -- get freqs for RoPE
        freqs_x = self.time_embed.prepare_freqs(
            (eN, N), x.device, x.dtype)  # NC, time_dim
        freqs_y = self.time_embed.prepare_freqs(
            (1, 1), x.device, x.dtype)  # NC, time_dim

        y = self.cls_token.repeat((B, 1, 1))

        if self.use_pos_embed:
            x = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)

        # -- concat query mask_token ys
        x = x.flatten(1, 2)  # B N E D -> B NE D
        x = torch.cat([y, x], dim=1)
        freqs_x = torch.cat([freqs_y, freqs_x], dim=0).to(x)

        # -- fwd prop
        for blk in self.reconstructor_blocks:
            x = blk(x, freqs_x)  # B, NC, D
            if blk.return_attention == True:
                return x

        # x = self.reconstructor_norm(x)

        # x = self.reconstructor_proj(x)

        return x


class EEGTransformer(nn.Module):
    """ EEG Transformer """

    def __init__(
        self,
        img_size=(54, 1024),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,  # PatchNormEmbed
        init_std=0.02,
        interpolate_factor=2.,
        return_attention_layer=-1,
        logdir=None,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num

        self.num_heads = num_heads

        # --
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # --

        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        # --
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                is_causal=False, use_rope=False, return_attention=(i+1) == return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))

        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        print("logdir: ", logdir)
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            filename = os.path.join(logdir, "EEGTransformer_encoder.txt")
            with open(filename, 'w') as f:
                f.write(f"img_size: {img_size}\n")
                f.write(f"embed_dim: {embed_dim}\n")
                f.write(f"embed_num: {embed_num}\n")
                f.write(f"patch_embed: {self.patch_embed}\n")
                f.write(f"chan_embed: {self.chan_embed}\n")
                f.write(f"summary_token: {self.summary_token}\n")
                f.write(f"num_patches: {self.num_patches}\n")

    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT, ch
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'chan_embed', 'summary_token'}

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T
        # mask_x.shape mN, mC
        # mask_t.shape mN

        # -- patchify x
        x = self.patch_embed(x)
        B, N, C, D = x.shape

        assert N == self.num_patches[1] and C == self.num_patches[
            0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"

        if chan_ids is None:
            chan_ids = torch.arange(0, C)
            # print("chan_ids = torch.arange(0, C): ", chan_ids)
        chan_ids = chan_ids.to(x)
        # print("chan_ids.to(x): ", chan_ids)

        # -- add channels positional embedding to x
        # (1,C) -> (1,1,C,D)
        # print("self.chan_embed: ", self.chan_embed)
        # print("chan_ids: ", chan_ids)
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)  # B, mN, mC, D
            B, N, C, D = x.shape

        x = x.flatten(0, 1)  # BmN, mC, D

        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)  # BmN, mC+embed_num, D

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # B*N, mC+1, D
            if blk.return_attention == True:
                return x

        x = x[:, -summary_token.shape[1]:, :]

        if self.norm is not None:
            x = self.norm(x)

        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)  # B, mN, D

        x = x.reshape((B, N, self.embed_num, -1))

        return x


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGPTClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=22,
                 img_size=[58, 2000],
                 patch_stride=64,
                 use_channels_names=None,
                 use_mean_pooling=True,
                 norm_layer=nn.LayerNorm,
                 use_chan_conv=False,
                 max_norm_chan_conv=1,
                 logdir=None,
                 ** kwargs):

        super().__init__()

        self.use_chan_conv = use_chan_conv
        if use_chan_conv:

            self.chan_conv = torch.nn.Sequential(
                Conv2dWithConstraint(in_channels, img_size[0], 1),
                nn.BatchNorm2d(img_size[0]),
                nn.GELU(),
                nn.Conv2d(img_size[0], img_size[0], kernel_size=(
                    1, 55), groups=img_size[0], padding='same'),
                nn.BatchNorm2d(img_size[0]),
                nn.Dropout(0.8),
                # nn.Dropout(0.25),
            )

        target_encoder = EEGTransformer(
            img_size=img_size,
            patch_size=32*2,
            patch_stride=patch_stride,
            embed_dim=512,
            embed_num=4,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            logdir=logdir)

        print("made target_encoder")
        reconstructor = EEGTransformerReconstructor(
            num_patches=target_encoder.num_patches,
            patch_size=32*2,
            embed_dim=512,
            embed_num=4,
            reconstructor_embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        print("made reconstructor")
        self.target_encoder = target_encoder
        self.reconstructor = reconstructor
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)

        embed_dim = 512
        self.embed_dim = embed_dim
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Sequential(
            nn.Dropout(0.8),
            LinearWithConstraint(32768, num_classes),
        )

        print("Created chan_conv with weights of shape:",
              self.chan_conv[0].weight.shape, flush=True)

    def get_num_layers(self):
        return self.target_encoder.get_num_layers() + self.reconstructor.get_num_layers()

    def get_classifier(self):
        return self.head

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(["target_encoder."+x for x in self.target_encoder.no_weight_decay()] +
                   ["reconstructor."+x for x in self.reconstructor.no_weight_decay()])

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head_0 = nn.Linear(
            self.embed_dim, 22) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(
            22*31, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, chan_ids=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        if chan_ids is None:
            chan_ids = self.chans_id

        if self.use_chan_conv:
            x = x[:, :, None]
            x = self.chan_conv(x)[:, :, 0]

        x = self.target_encoder(x, chan_ids.to(x))
        return x

    def forward(self, x, chan_ids=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, Times]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4*256]
        '''
        if len(x.shape) == 4:
            x = x.flatten(2)

        x = self.forward_features(
            x, chan_ids=chan_ids, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        # print(x.shape)

        # x = x.flatten(2)
        # x = x[:,:,0]
        # x = self.act(self.head_0(x))

        x = x.flatten(1)
        x = self.head(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = False):
        return super().load_state_dict(state_dict, strict)
    
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        # param.ndim <= 1 len(param.shape) == 1
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(
        ), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.learning_rate, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum,
                         nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.learning_rate:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9,
                              momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer



if __name__ == '__main__':

    args = Namespace(
    batch_size=8,
    epochs=10,
    learning_rate=0.00003,
    log_dir="./",
    output_dir="./",
    finetune="/Users/cccohen/deepEEG/unified/eegpt_mcae_58chs_4s_large4E.ckpt",
    seed=0,
    use_mean_pooling=True,
    input_size=200,
    update_freq=1,
    layer_decay=0.65,
    weight_decay_end=None,
    opt="adamw",
    weight_decay=0.05,
    min_lr=1e-6,
    warmup_epochs=5,
    warmup_steps=-1,
    start_epoch=0,
    clip_grad=5.0,
    hpc=True
)
    
    if args.hpc:
        args.finetune = "/home/chntzi001/deepEEG/EEGPT/downstream/Checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
        args.batch_size = 256
        args.log_dir= f"/home/chntzi001/deepEEG/EEGPT/downstream/log/test"
        args.output_dir = f"/scratch/chntzi001/khula/checkpoints/finetune_khula_eegpt/test"
        

    main(args)
