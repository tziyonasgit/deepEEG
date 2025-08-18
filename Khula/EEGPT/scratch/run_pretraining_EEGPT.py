import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import wandb
import csv
import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from engine_for_pretraining_EEGPT import train_one_epoch, evaluate, getVarOutputs
from utils import NativeScalerWithGradNormCount as NativeScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from pyexpat import model
from engine_pretraining import *


def get_args():
    parser = argparse.ArgumentParser(
        'pretraining and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--log_dir', default="./log", type=str)
    parser.add_argument(
        '--output_dir', default="/scratch/chntzi001/khula/checkpoints/pretrain_khula_eegpt", type=str)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias',
                        action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias',
                        action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_pretrain',
                        action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=5.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=4, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true', default=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--dataset', default='KHULA', type=str,
                        help='dataset: TUAB | TUEV')

    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias',
                        action='store_true', default=False)
    parser.add_argument('--sweep', default=True)
    parser.add_argument('--samplelength', default=4, type=int)
    known_args, _ = parser.parse_known_args()
    ds_init = None

    return parser.parse_args(), ds_init


def get_models(args, num_training_steps_per_epoch):
    global configs

    use_channels_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2',
                          'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
    ch_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4',
                'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

    if args.samplelength == 4:
        timepoints = 1024
    else:
        timepoints = 2560

    global modeltype
    modeltype = "tiny1"
    configs = {
        "embed_dim": 64, "embed_num": 1, "depth": [2, 2, 4], "num_heads": 4}

    model = LitEEGPT(configs, args.epochs, timepoints,
                     num_training_steps_per_epoch)

    print("This is the head of the model:", model.get_classifier())

    return model


def get_dataset(args):
    if args.dataset == 'KHULA':
        print("Preparing KHULA dataset...")
        if args.nb_classes > 1:
            filepath = "/scratch/chntzi001/khula/processed/"
            args.nb_classes = 4
            metrics = ["accuracy", "balanced_accuracy",
                       "cohen_kappa", "f1_weighted"]
        else:
            filepath = "/scratch/chntzi001/khula/processedBinary/"
            args.nb_classes = 1
            metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        train_dataset, test_dataset, val_dataset = utils.prepare_KHULA_dataset(
            filepath, args.nb_classes)

        ch_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4',
                    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def write_args_to_file(args, output_dir, run_name):
    args_file = os.path.join(output_dir, "args.txt")
    with open(args_file, "w") as f:
        f.write(
            f"date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"name of run: {run_name}\n")
        f.write(f"sample length: {args.samplelength}\n")
        f.write(f"----------- main hyperparameters: -----------\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"update_freq: {args.update_freq}\n")
        f.write(f"save_ckpt_freq: {args.save_ckpt_freq}\n")
        f.write(f"opt: {args.opt}\n")
        f.write(f"opt_eps: {args.opt_eps}\n")
        f.write(f"opt_betas: {args.opt_betas}\n")
        f.write(f"clip_grad: {args.clip_grad}\n")
        f.write(f"momentum: {args.momentum}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"weight_decay_end: {args.weight_decay_end}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"layer_decay: {args.layer_decay}\n")
        f.write(f"warmup_lr: {args.warmup_lr}\n")
        f.write(f"min_lr: {args.min_lr}\n")
        f.write(f"warmup_epochs: {args.warmup_epochs}\n")
        f.write(f"warmup_steps: {args.warmup_steps}\n")
        f.write(f"--------------------------------------------\n")
        f.write(f"robust_test: {args.robust_test}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"qkv_bias: {args.qkv_bias}\n")
        f.write(f"rel_pos_bias: {args.rel_pos_bias}\n")
        f.write(f"abs_pos_emb: {args.abs_pos_emb}\n")
        f.write(f"layer_scale_init_value: {args.layer_scale_init_value}\n")
        f.write(f"input_size: {args.input_size}\n")
        f.write(f"drop: {args.drop}\n")
        f.write(f"attn_drop_rate: {args.attn_drop_rate}\n")
        f.write(f"drop_path: {args.drop_path}\n")
        f.write(f"smoothing: {args.smoothing}\n")
        f.write(f"reprob: {args.reprob}\n")
        f.write(f"remode: {args.remode}\n")
        f.write(f"recount: {args.recount}\n")
        f.write(f"resplit: {args.resplit}\n")
        f.write(f"use_mean_pooling: {args.use_mean_pooling}\n")
        f.write(
            f"disable_weight_decay_on_rel_pos_bias: {args.disable_weight_decay_on_rel_pos_bias}\n")
        f.write(f"nb_classes: {args.nb_classes}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"log_dir: {args.log_dir}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"save_ckpt: {args.save_ckpt}\n")
        f.write(f"start_epoch: {args.start_epoch}\n")
        f.write(f"eval: {args.eval}\n")
        f.write(f"dist_eval: {args.dist_eval}\n")
        f.write(f"num_workers: {args.num_workers}\n")
        f.write(f"pin_mem: {args.pin_mem}\n")
        f.write(f"dataset: {args.dataset}\n")


def add_args_to_file(output_dir, line):
    args_file = os.path.join(output_dir, "args.txt")
    with open(args_file, "a") as f:
        f.write(f"{line}\n")


def main(args, ds_init):

    if args.sweep:
        global run_name
        wandb.init(
            project="deepEEG",
            config=vars(args)
        )
        config = wandb.config
        args.seed = config.seed
        run_name = f"lr{args.lr:.5f}_bs{args.batch_size}_seed{args.seed}"
        wandb.run.name = run_name
    else:
        print(f"lr{args.lr:.6f}@!!!!!!!")
        run_name = f"lr{args.lr:.5f}_bs{args.batch_size}_seed{args.seed}"

    if args.nb_classes > 1:
        classification = "multiclass"
    else:
        classification = "binary"

    args.output_dir = f"/scratch/chntzi001/khula/checkpoints/pretrain_khula_eegpt/{classification}/15-08/{run_name}"
    output_dir = args.output_dir
    args.log_dir = f"/home/chntzi001/deepEEG/EEGPT/pretrain/log/{classification}/15-08/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    tsneplots = f"/scratch/chntzi001/khula/checkpoints/pretrain_khula_eegpt/{classification}/15-08/{run_name}/tsne_plots"
    os.makedirs(tsneplots, exist_ok=True)
    write_args_to_file(args, output_dir, run_name)

    hyperparameters = {
        "model": modeltype,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }

    penultimate_layer = "(fc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)"

    output_csv = os.path.join(args.log_dir, "predictions.csv")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "True Label"])

    cluster_csv = os.path.join(tsneplots, "clusterdist.csv")
    with open(cluster_csv, mode='w') as file:
        writer = csv.writer(file)
        if args.nb_classes > 1:
            writer.writerow(['Cluster', '3M', '6M', '12M', '24M'])
        else:
            writer.writerow(['Cluster', '3M', '24M'])

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(
        args)

    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if not args.eval and torch.cuda.is_available():
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.SequentialSampler(
                dataset_train)
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval and dataset_val is not None:
            print("Setting up distributed evaluation")
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if dataset_test is not None:
                if type(dataset_test) == list:
                    sampler_test = [torch.utils.data.DistributedSampler(
                        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
                else:
                    sampler_test = torch.utils.data.DistributedSampler(
                        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            print("got here.....")
            print("setting up sampler val and test")
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    print("Setting up training dataset")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        if type(dataset_test) == list:
            print("dataset_test is a list")
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
            print("data_loader_test type:", type(data_loader_test))
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    model = get_models(args, num_training_steps_per_epoch)
    add_args_to_file(output_dir, configs)
    theOutputs = getVarOutputs()
    h = model.target_encoder.norm.register_forward_hook(
        getOutput('penultimate', theOutputs))

    patch_size = 64  # model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    model.to(device)

    model_ema = None
    model_without_ddp = model
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training per epoch = %d" %
          num_training_steps_per_epoch)

    add_args_to_file(output_dir, "--------------------------")
    line = "Number of training examples = %d" % len(dataset_train)
    add_args_to_file(output_dir, line)
    line = "Number of training per epoch = %d" % num_training_steps_per_epoch
    add_args_to_file(output_dir, line)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(
                "blocks.%d.attn.relative_position_bias_table" % i)

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    # selects loss function
    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        print("Using CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        print("$$$$$$$$$$$$$$$$$$$ Performing evaluation only, so skipping training $$$$$$$$$$$$$$$$$$$")
        balanced_accuracy = []
        accuracy = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, args, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(
            f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)
    print("ch_names before training = %s" % str(ch_names))
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(
                epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats, feats, lbls = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_classes == 1,
        )

        # Creating TSNE plots
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        if isinstance(feats, list):
            feats = torch.cat(feats, dim=0)
        if isinstance(lbls, list):
            lbls = torch.cat(lbls, dim=0)
        features_2d = tsne.fit_transform(feats.cpu().numpy())
        unique_labels = np.unique(lbls.cpu().numpy())
        age = {0: '3M', 1: '6M', 2: '12M', 3: '24M'}
        colours = ['#800080',
                   "#5959EDFF",
                   "#0ABA0A",
                   '#000000']
        kmeans = KMeans(n_clusters=4, random_state=0)
        clusters = kmeans.fit_predict(features_2d)

        plt.figure(figsize=(8, 6))
        # cluster-wise colouring
        for cluster_id in np.unique(clusters):
            idx = clusters == cluster_id
            plt.scatter(
                features_2d[idx, 0], features_2d[idx, 1],
                color=colours[cluster_id],
                s=20, alpha=0.7,
                label=f"Cluster {cluster_id}"
            )
        plt.title("k=4")
        plt.legend(title="Cluster ID", loc="best")
        plt.grid(True)
        filename = f"tsne_epoch{epoch}.png"
        plotpath = os.path.join(tsneplots, filename)
        plt.savefig(plotpath, dpi=300, bbox_inches='tight')
        plt.close()

        # logging stats of clusters (via csv)
        cluster_ids = clusters
        true_labels = lbls.cpu().numpy()
        distribution = defaultdict(lambda: defaultdict(int))
        for cluster_id, true_label in zip(cluster_ids, true_labels):
            distribution[cluster_id][true_label] += 1

        with open(cluster_csv, mode='a') as file:
            writer = csv.writer(file)
            for cluster_id, label_counts in distribution.items():
                clusterID = f"{epoch}_{cluster_id}"
                writer.writerow([clusterID, label_counts.get(0, 0), label_counts.get(1, 0),
                                 label_counts.get(2, 0), label_counts.get(3, 0)])

        if args.sweep:
            wandb.log({f"train/{k}": v for k, v in train_stats.items()}
                      | {"epoch": epoch})

        if output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            print("============== Evaluating on validation and test set ==============")
            print("Here batch size is = %d" % int(1.5 * args.batch_size))
            val_stats = evaluate(data_loader_val, model, device, args, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)

            if args.sweep:
                wandb.log({f"val/{k}": v for k, v in val_stats.items()}
                          | {"epoch": epoch})
            print(
                f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test, model, device, args, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            if args.sweep:
                wandb.log({f"test/{k}": v for k, v in test_stats.items()}
                          | {"epoch": epoch})
            print(
                f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epochNum=epoch, epoch="best", model_ema=model_ema)
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

        textfile = f"log_lr{args.lr}_bs{args.batch_size}.txt"
        if output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, textfile), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    best = {
        "epoch": bestEpoch,
        "max_accuracy_val": max_accuracy,
        "max_accuracy_test": max_accuracy_test
    }

    with open(os.path.join(output_dir, textfile), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(hyperparameters) + "\n")
        f.write(json.dumps(best))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.sweep:
        wandb.finish()


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    print("lr!!!! is:", opts.lr)
    main(opts, ds_init)
