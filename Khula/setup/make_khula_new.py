from scipy.spatial.distance import cdist
from mne.io import read_raw_eeglab
import mne
import numpy as np
import os
import pickle
import re

import multiprocessing


drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF',
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['P1', 'PO1', 'F8', 'C2', 'CZ', 'PO2', 'FPZ', 'F3', 'CP4', 'CP3', 'PO3', 'C5', 'FC6', 'PO10', 'FP2', 'FC4', 'FT7', 'PO8', 'CP5', 'F2', 'P4', 'AFZ', 'P6', 'O2', 'P2', 'FC5', 'FC1', 'TP9', 'T7', 'C4',
                    'P8', 'T8', 'OZ', 'AF4', 'CP1', 'FCZ', 'TP7', 'PO4', 'AF3', 'C3', 'O1', 'P7', 'F4', 'F1', 'FT8', 'CP2', 'CP6', 'PO7', 'P9', 'P5', 'P3', 'C6', 'PZ', 'FC2', 'PO9', 'POZ', 'C1', 'TP8', 'FZ', 'F7', 'P10', 'TP10', 'FC3']
standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]

checkpointChannels = ['FP1', 'FPZ', 'FP2',
                      'AF3', 'AF4',
                      'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                      'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                      'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                      'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                      'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
                      'O1', 'OZ', 'O2', ]

acceptedChannels = set([
    "PO12", "CCP2H", "FFC5H", "OI1", "PO7", "CPPZ",
    "TP7", "PO2", "FC3", "FTT7H", "PPO8", "CCP4H",
    "P11", "FCC5H", "FFC4H", "FP1", "CPP2H", "FFT7H",
    "P1", "I2", "AFF6H", "FZ", "PO4", "FCC2H",
    "F8", "FT9", "CP2", "AF3", "FCZ", "POO11H",
    "FPZ", "F3", "P8", "FC2", "F1", "CCP3H",
    "CP6", "PO1", "C1", "AFZ", "C3", "CB1",
    "FTT8H", "POO12H", "TP9", "I1", "FP2", "POO10H",
    "CPP1H", "CPP4H", "TTP8H", "AFF5H", "PO10", "POO9H",
    "POO3", "CP5", "PO3", "FC6", "FTT9H", "PPOZ",
    "TPP5H", "POO4", "CB2", "FT7", "CPZ", "CP1",
    "PPO1", "CP3", "CCP5H", "O2", "FCC1H", "CP4",
    "FT8", "T9", "PO5", "P2", "P5", "POZ",
    "FC1", "CPP3H", "C5", "P9", "P10", "PO6",
    "FFT8H", "CCP1H", "C2", "POOZ", "T7", "POO7",
    "FFC3H", "F6", "FCCZ", "TPP8H", "F7", "P4",
    "P3", "AF8", "PPO2", "AF4", "FFC2H", "FFC1H",
    "P6", "F2", "C6", "P12", "TP10", "CZ",
    "IZ", "CCP6H", "TP8", "PO11", "OI2", "FC5",
    "TTP7H", "CPP5H", "F5", "POO8", "CPP6H", "OZ",
    "PO9", "AF7", "PZ", "O1", "FC4", "PO8",
    "F4", "FCC3H", "T10", "P7", "FT10", "FCC4H",
    "FCC6H", "T8", "PPO7", "C4", "FFC6H", "FTT10H"
])


is_raw = False


def getChannelMapping():
    egi_montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    std1020_montage = mne.channels.make_standard_montage('standard_1020')

    # Get channel positions
    egi_pos = egi_montage.get_positions()['ch_pos']
    std_pos = std1020_montage.get_positions()['ch_pos']

    # Convert to arrays and label lists
    egi_labels, egi_xyz = zip(
        *[(k, v) for k, v in egi_pos.items() if k.startswith('E')])
    std_labels, std_xyz = zip(*std_pos.items())

    # Compute distance matrix
    distances = cdist(np.array(egi_xyz), np.array(std_xyz))

    # Find closest 10-20 match for each EGI channel
    egiTostandard = {}
    for i, label in enumerate(egi_labels):
        closest_idx = np.argmin(distances[i])
        closest_label = std_labels[closest_idx]
        egiTostandard[label] = closest_label

    return egiTostandard


def split_and_dump(params):
    full_file_path, dump_folder, label, chOrder_standard, egiStandardDict = params
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

    # set the montage
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    data.set_montage(montage)

    # print("the dictionary:", egiStandardDict)

    # filter dictionary
    available_chs = data.info['ch_names']
    filtered_mapping = {k.upper(): v.upper() for k,
                        v in egiStandardDict.items() if k in available_chs}

    # remove duplicate targets
    used_targets = set()
    unique_mapping = {}
    for k, v in filtered_mapping.items():
        if v not in used_targets:
            unique_mapping[k] = v
            used_targets.add(v)

    # print("new dictionary: ", unique_mapping)

    # print("Channel names before renaming:", data.ch_names)
    data.rename_channels(unique_mapping)
    data.pick(list(unique_mapping.values()))
    # print("Final channels:", data.ch_names)

    # the dictionary: {'E1': 'AF8', 'E2': 'AF6', 'E3': 'AF4', 'E4': 'F2', 'E5': 'Fz', 'E6': 'FCz', 'E7': 'FCz', 'E8': 'Fp2', 'E9': 'Fp2', 'E10': 'AF2', 'E11': 'AFz', 'E12': 'Fz', 'E13': 'FC1', 'E14': 'Fpz', 'E15': 'Fpz', 'E16': 'AFz', 'E17': 'Fpz', 'E18': 'AF1', 'E19': 'F1', 'E20': 'F1', 'E21': 'Fpz', 'E22': 'AF3', 'E23': 'AF3', 'E24': 'F3', 'E25': 'Fp1', 'E26': 'AF5', 'E27': 'F3', 'E28': 'F3', 'E29': 'FC3', 'E30': 'FC1', 'E31': 'C1', 'E32': 'AF7', 'E33': 'F7', 'E34': 'FC5', 'E35': 'FC3', 'E36': 'C3', 'E37': 'C1', 'E38': 'F9', 'E39': 'FT7', 'E40': 'C5', 'E41': 'C3', 'E42': 'C3', 'E43': 'F9', 'E44': 'FT9', 'E45': 'T7', 'E46': 'C5', 'E47': 'CP5', 'E48': 'F9', 'E49': 'FT9', 'E50': 'TP7', 'E51': 'CP5', 'E52': 'CP3', 'E53': 'CP1', 'E54': 'CP1', 'E55': 'Cz', 'E56': 'A1', 'E57': 'TP9', 'E58': 'P7', 'E59': 'P5', 'E60': 'P3', 'E61': 'P1', 'E62': 'Pz', 'E63': 'M1', 'E64': 'P9', 'E65': 'PO7', 'E66': 'PO3', 'E67': 'P1', 'E68': 'P9', 'E69': 'PO9', 'E70': 'O1', 'E71': 'PO1', 'E72': 'POz', 'E73': 'PO9', 'E74': 'O9', 'E75': 'Oz', 'E76': 'PO2', 'E77': 'P2', 'E78': 'P2', 'E79': 'CP2', 'E80': 'C2', 'E81': 'Iz', 'E82': 'O10', 'E83': 'O2', 'E84': 'PO4', 'E85': 'P4', 'E86': 'CP2', 'E87': 'C2', 'E88': 'PO10', 'E89': 'PO10', 'E90': 'PO8', 'E91': 'P6', 'E92': 'CP4', 'E93': 'C4', 'E94': 'P10', 'E95': 'P10', 'E96': 'P8', 'E97': 'CP6', 'E98': 'CP6', 'E99': 'M2', 'E100': 'TP10', 'E101': 'TP8', 'E102': 'C6', 'E103': 'C4', 'E104': 'C4', 'E105': 'FC2', 'E106': 'FCz', 'E107': 'A2', 'E108': 'T8', 'E109': 'C6', 'E110': 'FC4', 'E111': 'FC4', 'E112': 'FC2', 'E113': 'FT10', 'E114': 'FT10', 'E115': 'FT8', 'E116': 'FC6', 'E117': 'F4', 'E118': 'F2', 'E119': 'F10', 'E120': 'F10', 'E121': 'F10', 'E122': 'F8', 'E123': 'F4', 'E124': 'F4', 'E125': 'F10', 'E126': 'AF10', 'E127': 'AF9', 'E128': 'F9'}
    # new dictionary:  {'E2': 'AF6', 'E3': 'AF4', 'E4': 'F2', 'E5': 'FZ', 'E6': 'FCZ', 'E9': 'FP2', 'E10': 'AF2', 'E11': 'AFZ', 'E13': 'FC1', 'E15': 'FPZ', 'E18': 'AF1', 'E19': 'F1', 'E22': 'AF3', 'E24': 'F3', 'E26': 'AF5', 'E29': 'FC3', 'E31': 'C1', 'E33': 'F7', 'E34': 'FC5', 'E36': 'C3', 'E39': 'FT7', 'E40': 'C5', 'E45': 'T7', 'E47': 'CP5', 'E50': 'TP7', 'E52': 'CP3', 'E53': 'CP1', 'E55': 'CZ', 'E57': 'TP9', 'E58': 'P7', 'E59': 'P5', 'E60': 'P3', 'E61': 'P1', 'E62': 'PZ', 'E63': 'M1', 'E64': 'P9', 'E65': 'PO7', 'E66': 'PO3', 'E69': 'PO9', 'E70': 'O1', 'E71': 'PO1', 'E72': 'POZ', 'E74': 'O9', 'E75': 'OZ', 'E76': 'PO2', 'E77': 'P2', 'E79': 'CP2', 'E80': 'C2', 'E82': 'O10', 'E83': 'O2', 'E84': 'PO4', 'E85': 'P4', 'E89': 'PO10', 'E90': 'PO8', 'E91': 'P6', 'E92': 'CP4', 'E93': 'C4', 'E95': 'P10', 'E96': 'P8', 'E97': 'CP6', 'E100': 'TP10', 'E101': 'TP8', 'E102': 'C6', 'E105': 'FC2', 'E108': 'T8', 'E110': 'FC4', 'E115': 'FT8', 'E116': 'FC6', 'E117': 'F4', 'E122': 'F8'}
    # Channel names before renaming: ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E39', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124']
    # Channel names after renaming: ['AF6', 'AF4', 'F2', 'FZ', 'FCZ', 'E7', 'FP2', 'AF2', 'AFZ', 'E12', 'FC1', 'FPZ', 'E16', 'AF1', 'F1', 'E20', 'AF3', 'E23', 'F3', 'AF5', 'E27', 'E28', 'FC3', 'E30', 'C1', 'F7', 'FC5', 'E35', 'C3', 'E37', 'FT7', 'C5', 'E41', 'E42', 'T7', 'E46', 'CP5', 'TP7', 'E51', 'CP3', 'CP1', 'E54', 'CZ', 'TP9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'M1', 'P9', 'PO7', 'PO3', 'E67', 'PO9', 'O1', 'PO1', 'POZ', 'O9', 'OZ', 'PO2', 'P2', 'E78', 'CP2', 'C2', 'O10', 'O2', 'PO4', 'P4', 'E86', 'E87', 'PO10', 'PO8', 'P6', 'CP4', 'C4', 'P10', 'P8', 'CP6', 'E98', 'TP10', 'TP8', 'C6', 'E103', 'E104', 'FC2', 'E106', 'T8', 'E109', 'FC4', 'E111', 'E112', 'FT8', 'FC6', 'F4', 'E118', 'F8', 'E123', 'E124']

    data_channels = set(data.info['ch_names'])
    missing_channels = data_channels - acceptedChannels

    # if not missing_channels:
    #     print("✅ All data channels are in the allowed list.")
    # else:
    #     print(
    #         f"❌ These data channels are NOT in the allowed list:\n{sorted(missing_channels)}")

    allowed_channels = data_channels - missing_channels
    data.pick(list(allowed_channels))

    data_channels = set(data.info['ch_names'])
    checkchannels = data_channels - acceptedChannels
    # if not checkchannels:
    #     print("✅ All data channels are in the allowed list.")
    # else:
    #     print(
    #         f"❌ These data channels are NOT in the allowed list:\n{sorted(missing_channels)}")

    # print("The number of channels: ", len(data_channels))
    # print("the channels are: ", data.ch_names)
    try:
        if drop_channels is not None:
            useless_chs = []
            for ch in drop_channels:
                if ch in data.ch_names:
                    useless_chs.append(ch)
            data.drop_channels(useless_chs)
        if chOrder_standard is not None and len(chOrder_standard) == len(data.ch_names):
            data.reorder_channels(chOrder_standard)

        if data.ch_names != chOrder_standard:
            raise Exception(
                f"channel order is wrong!\Got: {data.ch_names}\nexpected: {chOrder_standard}")

        data.filter(l_freq=0.1, h_freq=75.0)
        data.resample(256, n_jobs=5)

        # ch_name = data.ch_names
        # print("Channel names:", data.ch_names)

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
        print(f"Processed {file} with shape: {channeled_data.shape}")

        # channeled_data.shape[1] -> tells us number of time points
        # we should have 2000 time points in each recording
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
        print("!!!!!!!!done file: ", file)

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

    egiStandardDict = getChannelMapping()

    # root to dataset
    root = "/scratch/chntzi001/khula"

    # eeg recording file name e.g. 1_191_14933186_12_T_20230720_011457002_processed.set

    with open("train.txt") as f:
        train_files = [line.strip() for line in f]

    with open("val.txt") as f:
        val_files = [line.strip() for line in f]

    with open("test.txt") as f:
        test_files = [line.strip() for line in f]

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "processed")):
        os.makedirs(os.path.join(root, "processed"))

    if not os.path.exists(os.path.join(root, "processed", "train")):
        os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(root, "processed", "train")

    if not os.path.exists(os.path.join(root, "processed", "val")):
        os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(root, "processed", "val")

    if not os.path.exists(os.path.join(root, "processed", "test")):
        os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(root, "processed", "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []

    for train_sub in train_files:
        train_sub_src, train_dump_folder = splitSrcDest(train_sub)
        match = re.search(r'_(3|6|12|24)_', train_sub_src)
        parameters.append(
            [train_sub_src, train_dump_folder, match.group(1), chOrder_standard, egiStandardDict])

    for val_sub in val_files:
        val_sub_src, val_dump_folder = splitSrcDest(val_sub)
        match = re.search(r'_(3|6|12|24)_', val_sub_src)
        if match:
            session = match.group(1)
            parameters.append(
                [val_sub_src, val_dump_folder, session, chOrder_standard, egiStandardDict])
        else:
            raise ValueError(f"Failed to extract session from: {val_sub_src}")

    for test_sub in test_files:
        test_sub_src, test_dump_folder = splitSrcDest(test_sub)
        match = re.search(r'_(3|6|12|24)_', test_sub_src)
        parameters.append(
            [test_sub_src, test_dump_folder, match.group(1), chOrder_standard, egiStandardDict])

    # split and dump in parallel
    with multiprocessing.Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

    # # Use a loop:
    # for args in parameters:
    #     split_and_dump(args)
