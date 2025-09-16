""" 
Purpose: Make the synthetic classification dataset from sine waves
        
Author: Tziyona Cohen, UCT
"""
from scipy.spatial.distance import cdist
from mne.io import read_raw_eeglab
import mne
import numpy as np
import os
import pickle
import re
import multiprocessing

egi_to_1020 = {
    'E3': 'AF4',
    'E4': 'F2',
    'E6': 'FCZ',
    'E9': 'FP2',
    'E11': 'FZ',
    'E13': 'FC1',
    'E15': 'FPZ',
    'E19': 'F1',
    'E23': 'AF3',
    'E24': 'F3',
    'E28': 'FC5',
    'E29': 'FC3',
    'E30': 'C1',
    'E33': 'F7',
    'E34': 'FT7',
    'E36': 'C3',
    'E37': 'CP1',
    'E41': 'C5',
    'E42': 'CP3',
    'E45': 'T7',
    'E46': 'TP7',
    'E47': 'CP5',
    'E51': 'P5',
    'E52': 'P3',
    'E55': 'CPZ',
    'E58': 'P7',
    'E60': 'P1',
    'E62': 'PZ',
    'E65': 'PO7',
    'E67': 'PO3',
    'E70': 'O1',
    'E72': 'POZ',
    'E75': 'OZ',
    'E77': 'PO4',
    'E83': 'O2',
    'E85': 'P2',
    'E87': 'CP2',
    'E90': 'PO8',
    'E92': 'P4',
    'E93': 'CP4',
    'E97': 'P6',
    'E96': 'P8',
    'E98': 'CP6',
    'E102': 'TP8',
    'E103': 'C6',
    'E104': 'C4',
    'E105': 'C2',
    'E108': 'T8',
    'E111': 'FC4',
    'E112': 'FC2',
    'E116': 'FT8',
    'E117': 'FC6',
    'E122': 'F8',
    'E124': 'F4'
}

removeChannels = ['E2', 'E5', 'E7', 'E10', 'E12', 'E16', 'E18', 'E20', 'E22', 'E26', 'E27', 'E31', 'E35', 'E39', 'E40', 'E50', 'E53', 'E54', 'E57', 'E59', 'E61', 'E63', 'E64',
                  'E66', 'E69', 'E71', 'E74', 'E76', 'E78', 'E79', 'E80', 'E82', 'E84', 'E86', 'E89', 'E91', 'E95', 'E100', 'E101', 'E106', 'E109', 'E110', 'E115', 'E118', 'E123']


chOrder_standard = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4',
                    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

checkpointChannels = set(['FP1', 'FPZ', 'FP2',
                          'AF3', 'AF4',
                          'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                          'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                          'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                          'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                          'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                          'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
                          'O1', 'OZ', 'O2', ])

is_raw = False
train_length = 32968
val_length = 3856
test_length = 4326
sampling_rate = 256
seed = 0
num_channels = 54
labels = {0: "3", 1: "6", 2: "12", 3: "24"}


def generate_sine_class_signal(num_channels, num_timepoints, amp, phase, noise_std, cls):
    t = np.linspace(0, 1, num_timepoints)

    if cls == 0:
        freq = np.random.uniform(0.5, 4)
    elif cls == 1:
        freq = np.random.uniform(4, 8)
    elif cls == 2:
        freq = np.random.uniform(8, 13)
    else:
        freq = np.random.uniform(13, 30)

    signal = amp * np.sin(2 * np.pi * freq * t + phase)
    signal = signal + np.random.normal(0, noise_std, size=signal.shape)

    data = np.stack([signal + np.random.normal(0, noise_std,
                    size=signal.shape) for i in range(num_channels)], axis=0)
    return data


def create(foldername, num, dump_folder, noise_std):
    # Params for each class
    np.random.seed(seed)
    class_params = [
        {
            "amp": np.random.uniform(0.5, 2.0),
            "phase": np.random.uniform(0, np.pi),
        }
        for i in range(4)]

    samples_per_class = num//4
    print(class_params)
    for c in range(4):
        cls_params = class_params[c]
        for s in range(samples_per_class):
            dump_path = os.path.join(
                dump_folder, f"{c}_{s}.pkl"
            )

            pickle.dump(
                {"X": generate_sine_class_signal(num_channels, 1024,
                                                 amp=cls_params["amp"], phase=cls_params["phase"], noise_std=noise_std, cls=c), "y": labels[c]},
                open(dump_path, "wb"),)
            print("Label is: ", labels[c])
            print("Done the file! ", dump_path)


if __name__ == "__main__":

    # root to dataset
    root = "/scratch/chntzi001/khula"
    param = os.environ.get("NOISE_STD")
    noise_std = float(param)
    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, f"processedSynthetic_{noise_std}")):
        os.makedirs(os.path.join(root,  f"processedSynthetic_{noise_std}"))

    if not os.path.exists(os.path.join(root,  f"processedSynthetic_{noise_std}", "train")):
        os.makedirs(os.path.join(
            root,  f"processedSynthetic_{noise_std}", "train"))
    train_dump_folder = os.path.join(
        root,  f"processedSynthetic_{noise_std}", "train")

    if not os.path.exists(os.path.join(root,  f"processedSynthetic_{noise_std}", "val")):
        os.makedirs(os.path.join(
            root,  f"processedSynthetic_{noise_std}", "val"))
    val_dump_folder = os.path.join(
        root,  f"processedSynthetic_{noise_std}", "val")

    if not os.path.exists(os.path.join(root,  f"processedSynthetic_{noise_std}", "test")):
        os.makedirs(os.path.join(
            root,  f"processedSynthetic_{noise_std}", "test"))
    test_dump_folder = os.path.join(
        root,  f"processedSynthetic_{noise_std}", "test")

    train_length = 32968
    val_length = 3856
    test_length = 4326

    create("train", train_length, train_dump_folder, noise_std)
    create("val", val_length, val_dump_folder, noise_std)
    create("test", test_length, test_dump_folder, noise_std)
