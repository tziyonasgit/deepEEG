from scipy.spatial.distance import cdist
from mne.io import read_raw_eeglab
import mne
import numpy as np
import os
import pickle

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
    fetch_folder, sub, dump_folder, label, chOrder_standard, egiStandardDict = params
    # print("!!!!!!!chOrder_standard:", chOrder_standard)
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            # raw = mne.io.read_epochs_eeglab(file_path)
            raw = mne.io.read_epochs_eeglab(file_path)
            montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
            raw.set_montage(montage)
            # Build a valid mapping with unique new names
            alreadyMappedChannels = set()
            valid_mapping = {}
            for old_ch, new_ch in egiStandardDict.items():
                if new_ch not in alreadyMappedChannels:
                    valid_mapping[old_ch] = new_ch
                    alreadyMappedChannels.add(new_ch)

            existing_chs = set(raw.info['ch_names'])
            filtered_mapping = {}
            for old_ch, new_ch in valid_mapping.items():
                if old_ch in existing_chs:
                    filtered_mapping[old_ch] = new_ch.upper()

            raw.rename_channels(filtered_mapping)
            raw.pick(list(filtered_mapping.values()))
            print(raw.ch_names)
            print("filtered_mapping:", filtered_mapping)

            # pos = np.array([ch['loc'][:3]
            #                for ch in raw.info['chs'] if ch['loc'] is not None])
            # print("Max electrode radius (m):",
            #       np.max(np.linalg.norm(pos, axis=1)))
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                expected_order = list(filtered_mapping.values())
                if expected_order is not None and len(expected_order) == len(raw.ch_names):
                    raw.reorder_channels(expected_order)
                if raw.ch_names != expected_order:
                    raise Exception(
                        f"channel order is wrong!\nraw: {raw.ch_names}\nexpected: {filtered_mapping}")

                # raw.filter(l_freq=0.1, h_freq=75.0)
                # raw.notch_filter(50.0)
                # raw.resample(200, n_jobs=5)  # downsizes from 1000Hz to 200Hzs

                ch_name = raw.ch_names
                # raw_data = raw.get_data(units='uV')
                epochs_data = raw.get_data()  # shape: (n_epochs, n_channels, n_times)
                # shape: (n_channels, total_time)
                channeled_data = np.concatenate(epochs_data, axis=-1)
            except Exception as e:
                print(f"[ERROR] Processing failed for file: {file}")
                print(f"Reason: {e}")
                with open("khula-process-error-files.txt", "a") as f:
                    f.write(f"{file}: {str(e)}\n")
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                print(f"Saving pickle to {dump_path}")
                pickle.dump(
                    {"X": channeled_data[:, i *
                                         2000: (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    multiprocessing.set_start_method("spawn", force=True)

    egiStandardDict = getChannelMapping()
    # Standard channel order for EGI 128 channels
    chOrder_standard = [ch.upper() for ch in egiStandardDict.values()]
    # print("HEYYYYY chOrder_standard:", chOrder_standard)

    # root to abnormal dataset
    root = "/scratch/chntzi001/khula"

    # eeg recording file name e.g. 1_191_14933186_12_T_20230720_011457002_processed.set

    # train, val three subjects
    train_val_three = os.path.join(root, "train", "three")
    train_val_three_sub = list(
        set([item.split("_")[2] for item in os.listdir(train_val_three)])
    )
    np.random.shuffle(train_val_three_sub)
    train_three_sub, val_three_sub = (
        train_val_three_sub[: int(len(train_val_three_sub) * 0.8)],
        train_val_three_sub[int(len(train_val_three_sub) * 0.8):],
    )

    # train, val six months subjects
    train_val_six = os.path.join(root, "train", "six")
    train_val_six_sub = list(
        set([item.split("_")[2] for item in os.listdir(train_val_six)])
    )
    np.random.shuffle(train_val_six_sub)
    train_six_sub, val_six_sub = (
        train_val_six_sub[: int(len(train_val_six_sub) * 0.8)],
        train_val_six_sub[int(len(train_val_six_sub) * 0.8):],
    )

    # train, val twelve months subjects
    train_val_twelve = os.path.join(root, "train", "twelve")
    train_val_twelve_sub = list(
        set([item.split("_")[2] for item in os.listdir(train_val_twelve)])
    )
    np.random.shuffle(train_val_twelve_sub)
    train_twelve_sub, val_twelve_sub = (
        train_val_twelve_sub[: int(len(train_val_twelve_sub) * 0.8)],
        train_val_twelve_sub[int(len(train_val_twelve_sub) * 0.8):],
    )

    # train, val twentyfour months subjects
    train_val_twentyfour = os.path.join(
        root, "train", "twentyfour")
    train_val_twentyfour_sub = list(
        set([item.split("_")[2] for item in os.listdir(train_val_twentyfour)])
    )
    np.random.shuffle(train_val_twentyfour_sub)
    train_twentyfour_sub, val_twentyfour_sub = (
        train_val_twentyfour_sub[: int(len(train_val_twentyfour_sub) * 0.8)],
        train_val_twentyfour_sub[int(len(train_val_twentyfour_sub) * 0.8):],
    )

    # test three subjects
    test_three = os.path.join(root, "eval", "three")
    test_three_sub = list(set([item.split("_")[2]
                               for item in os.listdir(test_three)]))

    # test six subjects
    test_six = os.path.join(root, "eval", "six")
    test_six_sub = list(set([item.split("_")[2]
                             for item in os.listdir(test_six)]))

    # test twelve subjects
    test_twelve = os.path.join(root, "eval", "twelve")
    test_twelve_sub = list(set([item.split("_")[2]
                                for item in os.listdir(test_twelve)]))

    # test twentyfour subjects
    test_twentyfour = os.path.join(root, "eval", "twentyfour")
    test_twentyfour_sub = list(set([item.split("_")[2]
                                    for item in os.listdir(test_twentyfour)]))

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
    # TRAIN
    # train_params = []
    # for train_sub in train_three_sub:
    #     train_params.append(
    #         [train_val_three, train_sub, train_dump_folder, 3, chOrder_standard, egiStandardDict])
    # for train_sub in train_six_sub:
    #     train_params.append(
    #         [train_val_six, train_sub, train_dump_folder, 6, chOrder_standard, egiStandardDict])
    # for train_sub in train_twelve_sub:
    #     train_params.append(
    #         [train_val_twelve, train_sub, train_dump_folder, 12, chOrder_standard, egiStandardDict])
    # for train_sub in train_twentyfour_sub:
    #     train_params.append(
    #         [train_val_twentyfour, train_sub, train_dump_folder, 24, chOrder_standard, egiStandardDict])
    # np.random.shuffle(train_params)
    # parameters.extend(train_params[:10])

    # VAL
    # val_params = []
    # for val_sub in val_three_sub:
    #     val_params.append(
    #         [train_val_three, val_sub, val_dump_folder, 3, chOrder_standard, egiStandardDict])
    # for val_sub in val_six_sub:
    #     val_params.append(
    #         [train_val_six, val_sub, val_dump_folder, 6, chOrder_standard, egiStandardDict])
    # for val_sub in val_twelve_sub:
    #     val_params.append([train_val_twelve, val_sub,
    #                       val_dump_folder, 12, chOrder_standard, egiStandardDict])
    # for val_sub in val_twentyfour_sub:
    #     val_params.append([train_val_twentyfour, val_sub,
    #                       val_dump_folder, 24, chOrder_standard, egiStandardDict])
    # np.random.shuffle(val_params)
    # parameters.extend(val_params[:10])

    # TEST
    # test_params = []
    # for test_sub in test_three_sub:
    #     test_params.append(
    #         [test_three, test_sub, test_dump_folder, 3, chOrder_standard, egiStandardDict])
    # for test_sub in test_six_sub:
    #     test_params.append(
    #         [test_six, test_sub, test_dump_folder, 6, chOrder_standard, egiStandardDict])
    # for test_sub in test_twelve_sub:
    #     test_params.append(
    #         [test_twelve, test_sub, test_dump_folder, 12, chOrder_standard, egiStandardDict])
    # for test_sub in test_twentyfour_sub:
    #     test_params.append(
    #         [test_twentyfour, test_sub, test_dump_folder, 24, chOrder_standard, egiStandardDict])
    # np.random.shuffle(test_params)
    # parameters.extend(test_params[:10])

    for train_sub in train_three_sub:
        parameters.append(
            [train_val_three, train_sub, train_dump_folder, 3, chOrder_standard, egiStandardDict])
    for train_sub in train_six_sub:
        parameters.append(
            [train_val_six, train_sub, train_dump_folder, 6, chOrder_standard, egiStandardDict])
    for train_sub in train_twelve_sub:
        parameters.append([train_val_twelve, train_sub,
                          train_dump_folder, 12, chOrder_standard, egiStandardDict])
    for train_sub in train_twentyfour_sub:
        parameters.append(
            [train_val_twentyfour, train_sub, train_dump_folder, 24, chOrder_standard, egiStandardDict])

    for val_sub in val_three_sub:
        parameters.append(
            [train_val_three, val_sub, val_dump_folder, 3, chOrder_standard, egiStandardDict])
    for val_sub in val_six_sub:
        parameters.append(
            [train_val_six, val_sub, val_dump_folder, 6, chOrder_standard, egiStandardDict])
    for val_sub in val_twelve_sub:
        parameters.append([train_val_twelve, val_sub,
                          val_dump_folder, 12, chOrder_standard, egiStandardDict])
    for val_sub in val_twentyfour_sub:
        parameters.append([train_val_twentyfour, val_sub,
                          val_dump_folder, 24, chOrder_standard, egiStandardDict])

    for test_sub in test_three_sub:
        parameters.append(
            [test_three, test_sub, test_dump_folder, 3, chOrder_standard, egiStandardDict])
    for test_sub in test_six_sub:
        parameters.append([test_six, test_sub, test_dump_folder,
                          6, chOrder_standard, egiStandardDict])
    for test_sub in test_twelve_sub:
        parameters.append([test_twelve, test_sub, test_dump_folder,
                          12, chOrder_standard, egiStandardDict])
    for test_sub in test_twentyfour_sub:
        parameters.append([test_twentyfour, test_sub,
                          test_dump_folder, 24, chOrder_standard, egiStandardDict])

    # split and dump in parallel
    with multiprocessing.Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

    # # Use a loop:
    # for args in parameters:
    #     split_and_dump(args)
