import os
from pathlib import Path
import pickle

from multiprocessing import Pool
import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold
import sys

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF',
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF', 'EEG 25+-REF', 'EEG 26+-REF', 'EEG 27+-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
# chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF',
#                   'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
chOrder_standard = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
    'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
    'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
]

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


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_fif(file_path, preload=True)
            raw.rename_channels(
                {ch: f'EEG {ch.upper()}-REF' for ch in raw.ch_names})
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                missing_channels = set(chOrder_standard) - set(raw.ch_names)
                if missing_channels:
                    print(
                        f"[ERROR] Processing failed for file: {Path(file).name}")
                    print(
                        f"Reason: channel order is wrong! Missing: {missing_channels}")
                    return  # EXIT EARLY before using undefined variables
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception(
                        "channel order is wrong!" + str(raw.ch_names))

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
                if channeled_data.shape[1] < 2000:
                    raise Exception(
                        f"[WARN] Not enough data to split: {Path(file).name}")
            except Exception as e:
                print(f"[ERROR] Processing failed for file: {file}")
                print(f"Reason: {e}")
                with open("quero-process-error-files.txt", "w") as f:
                    f.write(f"{file}: {str(e)}\n")

            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i *
                                         2000: (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


# returns subject's original path and its mapped label
def mapSub(subID):
    if subID in abSubjects:
        return abnormalFolder, 1
    else:
        return normalFolder, 0


if __name__ == "__main__":
    """
    Quero dataset is downloaded from https://openneuro.org/datasets/ds004577/versions/1.0.0
    Performing stratified k-fold cross validation
    """

    # root to dataset
    root = "/Users/cccohen/deepEEG"

    # collect subject IDs
    # /Users/cccohen/deepEEG/abnormal
    abnormalFolder = os.path.join(root, "abnormal")
    normalFolder = os.path.join(root, "normal")

    abSubjects = list(set([
        f.split(".")[0]
        for f in os.listdir(abnormalFolder)
        if os.path.isfile(os.path.join(abnormalFolder, f))
    ]))
    normSubjects = list(set([
        f.split(".")[0]
        for f in os.listdir(normalFolder)
        if os.path.isfile(os.path.join(normalFolder, f))
    ]))

    totalSubjects = abSubjects + normSubjects  # list of all subject IDs
    # list of all respective subject ID's label (1 - abnormal and 0 - normal)
    labels = [1] * len(abSubjects) + [0] * len(normSubjects)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # k = 5

    for foldCount, (trainCount, testCount) in enumerate(skf.split(totalSubjects, labels)):
        trainSubs = [totalSubjects[i] for i in trainCount]
        testSubs = [totalSubjects[i] for i in testCount]
        trainLabels = [labels[i] for i in trainCount]
        testLabels = [labels[i] for i in testCount]

        # create parameter list per fold
        # path name will be e.g. "/scratch/chntzi001/QUERO/processed/fold_1"
        foldDump = os.path.join(root, "processed", f"fold_{foldCount+1}")
        os.makedirs(foldDump, exist_ok=True)
        # path name will be e.g. "/scratch/chntzi001/QUERO/processed/fold_1/train"
        trainDump = os.path.join(foldDump, "train")
        testDump = os.path.join(foldDump, "test")
        os.makedirs(trainDump, exist_ok=True)
        os.makedirs(testDump, exist_ok=True)

        params = []
        for subID in trainSubs:
            folder, label = mapSub(subID)
            # adds an array of folder path, subject ID, train dump path, label to the list of training parameters
            params.append([folder, subID, trainDump, label])

        testParams = []
        for subID in testSubs:
            folder, label = mapSub(subID)
            # adds an array of folder path of subject's origin, subject ID, test dump path, label to the list of test parameters
            params.append([folder, subID, testDump, label])

        # split and dump in parallel
        with Pool(processes=24) as pool:
            # Use the pool.map function to apply the square function to each element in the numbers list
            result = pool.map(split_and_dump, params)


# # Use a loop:
# for args in parameters:
#     split_and_dump(args)
