from scipy.spatial.distance import cdist
from mne.io import read_raw_eeglab
from mne.time_frequency import psd_array_welch
import mne
import numpy as np
import os
import pickle
import re
import multiprocessing
import csv

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
features = {}
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def SUBID_to_label_mapping():
    bayley_csv = '/home/chntzi001/deepEEG/setUpKhula/bayley/khulasubsBayley.csv'

    with open(bayley_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subid = row["SUBID"]
            features[subid] = {
                "SUB_ID": subid,
                "COG": row["COG"],
                "LANG": row["LANG"],
                "MOT": row["MOT"],
                "3M_AGE": row["3M_AGE"],
                "6M_AGE": row["6M_AGE"],
                "12M_AGE": row["12M_AGE"],
                "24M_AGE": row["24M_AGE"],
            }


def split_and_dump(params):
    full_file_path, dump_folder, label = params
    file = os.path.basename(full_file_path)
    try:
        data = mne.io.read_epochs_eeglab(full_file_path)
        is_raw = False
    except ValueError as e:
        # Load as continuous data
        print(
            f"[WARNING] {file} has less than 2 trials, loading as continuous data.")
        data = mne.io.read_raw_eeglab(
            full_file_path, preload=True)
        is_raw = True
    except Exception as e2:
        print(f"[ERROR] Failed to read raw for {file}. Error: {e2}")
        with open("khula-process-error-files.txt", "a") as f:
            f.write(f"{file}: {str(e2)}\n")
        return  # skip this file

    data.drop_channels(removeChannels)
    data.rename_channels(egi_to_1020)
    data_channels = data.info['ch_names']

    try:
        data.reorder_channels(chOrder_standard)
        data_channels = data.info['ch_names']

        if data_channels != chOrder_standard:
            raise Exception(
                f"channel order is wrong!\Got:{data_channels}\nexpected: {chOrder_standard}")

        data.resample(256, n_jobs=5)
        spec = data.compute_psd(method="welch", fmin=0.4, fmax=55,
                                n_fft=512, n_overlap=256, picks="eeg")
        psds, freqs = spec.get_data(return_freqs=True)

        if not is_raw:
            # epochs -> shape: (n_epochs, n_channels, n_times)
            epochs_data = data.get_data()
            psds = psds.mean(axis=0)  # computes mean psd across epochs
            # final shape: (n_channels, n_times)

        bandpowers = {}
        for band, (lo, hi) in bands.items():
            keep = []
            for f in freqs:
                if (f >= lo) and (f < hi):
                    keep.append(True)
                else:
                    keep.append(False)
            bandpowers[band] = psds[:, keep].mean(axis=1)
            keep = []

        X = np.vstack(list(bandpowers.values())).T
        X = X.reshape(-1)  # to ensure it is a 1D feature

        dump_path = os.path.join(
            dump_folder, file.split(".")[0] + ".pkl"
        )
        pickle.dump(
            {"X": X, "y": label},
            open(dump_path, "wb"))
        print("The file is saved at:", dump_path)
        print("Done the file!!! ", file)

    except Exception as e:
        print(f"Processing failed for file: {file}")
        print(f"Reason: {e}")


def splitSrcDest(line):
    parts = line.split("_/scratch", 1)
    src = parts[0]
    dst = "/scratch" + parts[1]
    return src, dst


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    SUBID_to_label_mapping()

    # root to dataset
    root = "/scratch/chntzi001/khula"

    # eeg recording file name e.g. 1_191_14933186_12_T_20230720_011457002_processed.set

    with open("/home/chntzi001/deepEEG/linearReg/trainLinReg.txt") as f:
        train_files = [line.strip() for line in f]

    with open("/home/chntzi001/deepEEG/linearReg/testLinReg.txt") as f:
        test_files = [line.strip() for line in f]

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "processedLinReg")):
        os.makedirs(os.path.join(root, "processedLinReg"))

    if not os.path.exists(os.path.join(root, "processedLinReg", "train")):
        os.makedirs(os.path.join(root, "processedLinReg", "train"))
    train_dump_folder = os.path.join(root, "processedLinReg", "train")

    if not os.path.exists(os.path.join(root, "processedLinReg", "test")):
        os.makedirs(os.path.join(root, "processedLinReg", "test"))
    test_dump_folder = os.path.join(root, "processedLinReg", "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []

    for train_sub in train_files:
        train_sub_src, train_dump_folder = splitSrcDest(train_sub)
        subid = train_sub_src.split("_")[2]
        try:
            label = features[subid]
        except KeyError:
            print(f"Skipping {subid} bc not in features dictionary")
            continue
        parameters.append(
            [train_sub_src, train_dump_folder, label])

    for test_sub in test_files:
        test_sub_src, test_dump_folder = splitSrcDest(test_sub)
        subid = test_sub_src.split("_")[2]
        try:
            label = features[subid]
        except KeyError:
            print(f"Skipping {subid} bc not in feature dictionary")
            continue
        parameters.append(
            [test_sub_src, test_dump_folder, label])

    # split and dump in parallel
    with multiprocessing.Pool(processes=24) as pool:
        result = pool.map(split_and_dump, parameters)
