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
    print("renamed channels: ", data_channels)

    try:
        data.reorder_channels(chOrder_standard)
        data_channels = data.info['ch_names']

        if data_channels != chOrder_standard:
            raise Exception(
                f"channel order is wrong!\Got:{data_channels}\nexpected: {chOrder_standard}")

        data.filter(l_freq=49.0, h_freq=51.0)
        data.resample(256, n_jobs=5)

        print("The ACTUAL number of channels left: ", len(data.ch_names))
        print("the ACTUAL channels are: ", data.ch_names)
        print("final check of channel_10_20_names: ", chOrder_standard)

        if is_raw:
            # raw -> shape = (n_channels, n_times)
            raw_data = data.get_data(units='uV')
            channeled_data = raw_data.copy()
        else:
            # epochs -> shape: (n_epochs, n_channels, n_times)
            epochs_data = data.get_data()
            channeled_data = np.concatenate(epochs_data, axis=-1)
            # final shape: (n_channels, n_times)
        if channeled_data.ndim != 2 or channeled_data.shape[1] < 1024:
            raise ValueError(
                f"Invalid data shape: {channeled_data.shape} for file {file}")

        # channeled_data.shape[1] -> tells us number of time points
        for i in range(channeled_data.shape[1] // 1024):
            dump_path = os.path.join(
                dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
            )
            # print(f"Saving pickle to {dump_path}")
            pickle.dump(
                {"X": channeled_data[:, i *
                                     1024: (i + 1) * 1024], "y": label},
                open(dump_path, "wb"),
            )
            print("dump_path", dump_path)
        print("Done the file!!! ", file)

    except Exception as e:
        print(f"[ERROR] Processing failed for file: {file}")
        print(f"Reason: {e}")
        with open("khula-process-error-files.txt", "a") as f:
            f.write(f"{file}: {str(e)}\n")


def splitSrcDest(line):
    parts = line.split("_/scratch", 1)
    src = parts[0]
    dst = "/scratch" + parts[1]
    return src, dst


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # root to dataset
    root = "/scratch/chntzi001/khula"

    # eeg recording file name e.g. 1_191_14933186_12_T_20230720_011457002_processed.set

    with open("trainBinLog.txt") as f:
        train_files = [line.strip() for line in f]

    with open("testBinLog.txt") as f:
        test_files = [line.strip() for line in f]

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "processedBinLogReg")):
        os.makedirs(os.path.join(root, "processedBinLogReg"))

    if not os.path.exists(os.path.join(root, "processedBinLogReg", "train")):
        os.makedirs(os.path.join(root, "processedBinLogReg", "train"))
    train_dump_folder = os.path.join(root, "processedBinLogReg", "train")

    if not os.path.exists(os.path.join(root, "processedBinLogReg", "test")):
        os.makedirs(os.path.join(root, "processedBinLogReg", "test"))
    test_dump_folder = os.path.join(root, "processedBinLogReg", "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []

    label = None
    for train_sub in train_files:
        train_sub_src, train_dump_folder = splitSrcDest(train_sub)
        print(f"train_sub_src: {train_sub_src}")
        match = re.search(r'_(3|24)_', train_sub_src)
        if match.group(1) in ['3']:
            label = 0
        elif match.group(1) in ['24']:
            label = 1
        if label is not None:
            parameters.append(
                [train_sub_src, train_dump_folder, label])
        else:
            raise ValueError(
                f"Failed to extract session from: {train_sub_src}")

    for test_sub in test_files:
        test_sub_src, test_dump_folder = splitSrcDest(test_sub)
        match = re.search(r'_(3|24)_', test_sub_src)
        if match.group(1) in ['3']:
            label = 0
        elif match.group(1) in ['24']:
            label = 1
        if label is not None:
            parameters.append(
                [test_sub_src, test_dump_folder, label])
        else:
            raise ValueError(f"Failed to extract session from: {test_sub_src}")

    # split and dump in parallel
    with multiprocessing.Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

    # # Use a loop:
    # for args in parameters:
    #     split_and_dump(args)
