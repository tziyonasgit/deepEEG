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
from torcheeg.datasets import M3CVDataset, TSUBenckmarkDataset, DEAPDataset, SEEDDataset, moabb
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LIST, M3CV_CHANNEL_LIST, TSUBENCHMARK_CHANNEL_LIST
from torcheeg.datasets import CSVFolderDataset
from torchaudio.transforms import Resample

import glob
import warnings
warnings.filterwarnings(
    "ignore",
    message="At least one epoch has multiple events. Only the latency of the first event will be retained.",
    category=RuntimeWarning
)


# # current channel order for renamed khula dataset
# use_channels_names = ['AF6', 'AF4', 'F2', 'FZ', 'FCZ', 'AF2', 'AFZ', 'FC1', 'AF1', 'F1', 'AF3', 'F3', 'AF5', 'FC3', 'C1', 'F7', 'FC5', 'C3', 'FT7', 'C5', 'T7', 'CP5', 'TP7', 'CP3', 'CP1', 'CZ', 'TP9', 'P7', 'P5', 'P3', 'P1', 'PZ',
#                       'M1', 'P9', 'PO7', 'PO3', 'PO9', 'O1', 'PO1', 'POZ', 'O9', 'OZ', 'PO2', 'P2', 'CP2', 'C2', 'O10', 'O2', 'PO4', 'P4', 'PO8', 'P6', 'CP4', 'C4', 'P8', 'CP6', 'TP10', 'TP8', 'C6', 'FC2', 'T8', 'FC4', 'FT8', 'FC6', 'F4', 'F8']

KHULA_CHANNEL_LIST = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E39', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63',
                      'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124']


use_channels_names = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E39', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63',
                      'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124']


age_map = {3: 0, 6: 1, 12: 2, 24: 3}

# ------------------- PhysioMI

PHYSIONETMI_CHANNEL_LIST = ['Fc5.', 'Fc3.', 'Fc1.',
                            'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..',
                            'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.',
                            'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.',
                            'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..',
                            'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..',
                            'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..',
                            'Oz..', 'O2..', 'Iz..']
PHYSIONETMI_CHANNEL_LIST = [x.strip('.').upper()
                            for x in PHYSIONETMI_CHANNEL_LIST]


def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    print("CHECKING THAT I GOT HERE")
    print("the shape is: ", x.shape)
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError(
            "TemporalInterpolation only support sequence of single dim channels with optional batch")


def get_KHULA_dataset():

    # if not os.path.exists("/scratch/chntzi001/khula/io_root"):
    #     def default_read_fn(file_path, label_id=None, **kwargs):
    #         try:
    #             raw = mne.io.read_epochs_eeglab(file_path, verbose=False)
    #             return raw
    #         except ValueError as e:
    #             if "trials less than 2" in str(e):
    #                 print(f"⚠️ Skipping non-epoched file: {file_path}")
    #                 return None
    #             else:
    #                 raise  # Raise other unexpected errors

    #     print("got here!!!!")

    #     dataset = CSVFolderDataset(csv_path="/home/chntzi001/deepEEG/EEGPT/pretrain/khula_file_list.csv",
    #                                read_fn=default_read_fn,
    #                                io_path="/scratch/chntzi001/khula/io_root",
    #                                online_transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.To2d()
    #                                ]), label_transform=transforms.Select('label'),
    #                                num_worker=4)
    # else:
    dataset = CSVFolderDataset(
        io_path="/scratch/chntzi001/khula/io_root",
        online_transform=transforms.Compose([
                transforms.PickElectrode(transforms.PickElectrode.to_index_list(
                    use_channels_names, KHULA_CHANNEL_LIST)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: temporal_interpolation(
                    x, 1980) / 1000),  # V-> 1000uV and i changed it to resample to 1980 timepoints so it can be divided by 99
                transforms.To2d()
        ]),
        label_transform=transforms.Select('label'),
        num_worker=40
    )
    return dataset


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
        assert data.shape[1] == 1980
        print(i, data.shape, len(data.shape) ==
              2 and data.shape[0] == 99 and data.shape[1] >= 1000)
        assert len(
            data.shape) == 2 and data.shape[0] == 99 and data.shape[1] >= 1000
        torch.save(data, dst + tag+f"_{i}.set")
        del data, x
