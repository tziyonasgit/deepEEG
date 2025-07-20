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
chOrder_standard = []
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
    # print("!!!!!!!chOrder_standard:", chOrder_standard)
    # for file in os.listdir(fetch_file):
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

    # build a valid mapping with unique new names
    alreadyMappedChannels = set()
    valid_mapping = {}
    for old_ch, new_ch in egiStandardDict.items():
        if new_ch not in alreadyMappedChannels:
            valid_mapping[old_ch] = new_ch
            alreadyMappedChannels.add(new_ch)
    existing_chs = set(data.info['ch_names'])
    filtered_mapping = {}
    for old_ch, new_ch in valid_mapping.items():
        if old_ch in existing_chs:
            filtered_mapping[old_ch] = new_ch.upper()

    data.rename_channels(filtered_mapping)
    data.pick(list(filtered_mapping.values()))

    try:
        if drop_channels is not None:
            useless_chs = []
            for ch in drop_channels:
                if ch in data.ch_names:
                    useless_chs.append(ch)
            data.drop_channels(useless_chs)
        expected_order = list(filtered_mapping.values())
        if expected_order is not None and len(expected_order) == len(data.ch_names):
            data.reorder_channels(expected_order)

        if data.ch_names != expected_order:
            raise Exception(
                f"channel order is wrong!\Got: {data.ch_names}\nexpected: {filtered_mapping}")

        # current sampling frequency is 1000Hz

        # _______________________________________
        # raw.notch_filter(50.0)
        # _______________________________________
        data.filter(l_freq=0.1, h_freq=75.0)
        data.resample(200, n_jobs=5)  # downsizes from 1000Hz to 200Hzs

        # ch_name = data.ch_names
        print("Channel names:", data.ch_names)

        if is_raw:
            # raw -> shape = (n_channels, n_times)
            raw_data = data.get_data(units='uV')
            channeled_data = raw_data.copy()
        else:
            # epochs -> shape: (n_epochs, n_channels, n_times)
            epochs_data = data.get_data()
            channeled_data = np.concatenate(epochs_data, axis=-1)
            # final shape: (n_channels, n_times)
        if channeled_data.ndim != 2 or channeled_data.shape[1] < 2000:
            raise ValueError(
                f"Invalid data shape: {channeled_data.shape} for file {file}")
        print(f"Processed {file} with shape: {channeled_data.shape}")

        # if we did not resample then we have 2000/1000Hz = 2 second samples
        # now we will have 2000/200Hz = 10 seconds samples

        # channeled_data.shape[1] -> tells us number of time points
        # we should have 2000 time points in each recording
        for i in range(channeled_data.shape[1] // 2000):
            dump_path = os.path.join(
                dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
            )
            # print(f"Saving pickle to {dump_path}")
            pickle.dump(
                {"X": channeled_data[:, i *
                                     2000: (i + 1) * 2000], "y": label},
                open(dump_path, "wb"),
            )
            duration_sec = 2000 / data.info['sfreq']
            file_size = os.path.getsize(dump_path)

            print(
                f"[INFO] Saved: {dump_path} | Duration: {duration_sec:.2f} sec | Size: {file_size / 1024:.1f} KB")

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
    # Standard channel order for EGI 128 channels
    chOrder_standard = [ch.upper() for ch in egiStandardDict.values()]

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
        # print(
        #     f"train_sub_src: {train_sub_src}, match.group(1): {match.group(1)}")

    for val_sub in val_files:
        val_sub_src, val_dump_folder = splitSrcDest(val_sub)
        match = re.search(r'_(3|6|12|24)_', val_sub_src)
        if match:
            session = match.group(1)
            parameters.append(
                [val_sub_src, val_dump_folder, session, chOrder_standard, egiStandardDict])
            # print(
            #     f"val_sub_src: {val_sub_src}, session: {session}")
        else:
            raise ValueError(f"Failed to extract session from: {val_sub_src}")

    # for test_sub in test_six_sub:
    #     parameters.append([test_six, test_sub, test_dump_folder,
    #                       6, chOrder_standard, egiStandardDict])
    # for test_sub in test_twelve_sub:
    #     parameters.append([test_twelve, test_sub, test_dump_folder,
    #                       12, chOrder_standard, egiStandardDict])
    # for test_sub in test_twentyfour_sub:
    #     parameters.append([test_twentyfour, test_sub,
    #                       test_dump_folder, 24, chOrder_standard, egiStandardDict])
    for test_sub in test_files:
        test_sub_src, test_dump_folder = splitSrcDest(test_sub)
        match = re.search(r'_(3|6|12|24)_', test_sub_src)
        parameters.append(
            [test_sub_src, test_dump_folder, match.group(1), chOrder_standard, egiStandardDict])
        # print(
        #     f"test_sub_src: {test_sub_src}, match.group(1): {match.group(1)}")

    # split and dump in parallel
    with multiprocessing.Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

    # # Use a loop:
    # for args in parameters:
    #     split_and_dump(args)
