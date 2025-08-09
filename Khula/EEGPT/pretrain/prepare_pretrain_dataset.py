import os
import torch
import shutil
import random
import mne

import pandas as pd
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
import copy

import torcheeg
from torcheeg import transforms
from torcheeg.datasets import CSVFolderDataset
from torchaudio.transforms import Resample

import glob
import warnings
warnings.filterwarnings("ignore")

# KHULA_CHANNEL_LIST = ['FP1', 'FPZ', 'FP2',
#                       'AF3', 'AF4',
#                       'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#                       'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
#                       'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
#                       'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#                       'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
#                       'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
#                       'O1', 'OZ', 'O2', ]

use_channels_names = ['PZ', 'C2', 'P5', 'P6', 'TP8', 'C5', 'FC4', 'FT7', 'AF4', 'POZ', 'F6', 'TP7', 'PO7', 'PO4', 'O2', 'F8', 'F4', 'T7', 'CP6', 'PO8', 'C3', 'CP1', 'CP4', 'F3', 'OZ', 'FC3',
                      'FT8', 'F7', 'FP2', 'PO3', 'P4', 'F5', 'FC2', 'P2', 'AF3', 'CPZ', 'F2', 'CP5', 'FP1', 'FC1', 'P1', 'FZ', 'FPZ', 'CP3', 'O1', 'P3', 'C6', 'FC6', 'C4', 'F1', 'CP2', 'FCZ', 'FC5', 'C1']

egi_to_10_10 = {
    'E3': 'AF4', 'E4': 'F2', 'E6': 'FCZ', 'E9': 'FP2', 'E11': 'FZ', 'E13': 'FC1', 'E15': 'FPZ',
    'E19': 'F1', 'E22': 'FP1', 'E23': 'AF3', 'E24': 'F3',
    'E27': 'F5', 'E28': 'FC5', 'E29': 'FC3', 'E30': 'C1',
    'E33': 'F7', 'E34': 'FT7',  'E36': 'C3', 'E37': 'CP1', 'E41': 'C5', 'E42': 'CP3', 'E45': 'T7', 'E46': 'TP7', 'E47': 'CP5',
    'E51': 'P5', 'E52': 'P3',  'E55': 'CPZ',
    'E60': 'P1',  'E62': 'PZ',
    'E65': 'PO7',  'E67': 'PO3',
    'E70': 'O1', 'E72': 'POZ', 'E75': 'OZ',
    'E77': 'PO4', 'E83': 'O2',
    'E85': 'P2',  'E87': 'CP2', 'E90': 'PO8',
    'E92': 'P4', 'E93': 'CP4', 'E97': 'P6',
    'E98': 'CP6', 'E102': 'TP8', 'E103': 'C6',
    'E104': 'C4', 'E105': 'C2',
    'E111': 'FC4', 'E112': 'FC2',  'E116': 'FT8',
    'E117': 'FC6',  'E122': 'F8', 'E123': 'F6', 'E124': 'F4'
}

removeChannels = {'E100', 'E95', 'E64', 'E2', 'E57', 'E58', 'E96', 'E16', 'E26', 'E5', 'E7', 'E10', 'E18', 'E12', 'E20', 'E31', 'E39', 'E35', 'E40', 'E50', 'E53',
                  'E54', 'E59', 'E61', 'E63', 'E66', 'E69', 'E71', 'E74', 'E76', 'E78', 'E79', 'E80', 'E82', 'E84', 'E86', 'E89', 'E91', 'E101', 'E106', 'E108', 'E109', 'E110', 'E115', 'E118'}

age_map = {3: 0, 6: 1, 12: 2, 24: 3}


def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    print("the shape is: ", x.shape)
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        newData = torch.nn.functional.interpolate(x.unsqueeze(
            0), desired_sequence_length, mode=mode).squeeze(0)
        print("after interpolation shape: ", newData.shape)
        return newData
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError(
            "TemporalInterpolation only support sequence of single dim channels with optional batch")


def default_read_fn(file_path, label_id=None, **kwargs):
    print("hi we got here okay", flush=True)
    data = mne.io.read_epochs_eeglab(file_path, verbose=False)
    data.drop_channels(removeChannels)
    data.rename_channels(egi_to_10_10)
    print("Before reorder, channel names:", data.ch_names, flush=True)
    data.reorder_channels(use_channels_names)
    print("number of channels: ", len(data.ch_names), flush=True)
    if set(data.ch_names) != set(use_channels_names):
        missing = set(use_channels_names) - set(data.ch_names)
        print(f"❌ Missing channels: {missing} in {file_path}", flush=True)
        raise ValueError(f"Sample has missing channels: {missing}")
    data.filter(l_freq=0.1, h_freq=75.0)
    print("filtered...", flush=True)
    data.resample(256)
    print("Final channel names:", data.ch_names, flush=True)
    return data


def get_KHULA_dataset():
    io_root = "/scratch/chntzi001/khula/io_root"
    csv_path = "/home/chntzi001/deepEEG/EEGPT/pretrain/khula_file_list.csv"

    if os.path.exists(io_root):
        print("❗ Removing cached data to force reprocessing...")
        shutil.rmtree(io_root)

    try:
        print("📥 Reprocessing from scratch using read_fn")
        dataset = CSVFolderDataset(
            csv_path=csv_path,
            read_fn=default_read_fn,  # ✅ correct function name
            io_path=io_root,
            online_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: temporal_interpolation(x, 1024) / 1000),
                transforms.To2d()
            ]),
            label_transform=transforms.Select('label'),
            num_worker=1
        )
        return dataset
    except Exception as e:
        print(f"❌ Error occurred in CSVFolderDataset: {e}")
        raise  # Optional: re-raises for full traceback


if __name__ == "__main__":
    import random
    import os
    import tqdm

    tag = "khula"

    dataset = get_KHULA_dataset()

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][0].min(), dataset[0][0].max(),
          dataset[0][0].mean(), dataset[0][0].std())
    for i, (x, y) in tqdm.tqdm(enumerate(dataset)):
        dst = "/scratch/chntzi001/khula/pretrain/"
        label = int(y)
        print("counter is: ", {i})
        if random.random() < 0.1:
            dst += f"ValidFolder/0/"
        else:
            dst += f"TrainFolder/0/"
        os.makedirs(dst, exist_ok=True)
        data = x.squeeze_(0)
        # data = data.clone().detach().cpu()
        assert data.shape == (54, 1024), f"Wrong shape at {i}: {data.shape}"
        torch.save(data, dst + tag+f"_{i}.pt")
        del data, x
