import os
import shutil
import random
import re
from collections import defaultdict
import csv

# Path to raw EEG files (e.g. .set, .edf)
input_dir = '/scratch/chntzi001/khula'

output_root = '/scratch/chntzi001/khula/processedBinLinReg/'

months = [3, 24]
subject_files = defaultdict(list)
pattern = re.compile(r'\d+_\d+_(\d+)_(\d+)(?:_[ST])?_\d+_\d+_processed\.set')


with open('/home/chntzi001/deepEEG/setUpKhula/bayley/khulasubsBayley.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    firstRow = next(reader)  # Skip header
    all_rows = list(reader)
    for i, month in enumerate(months):
        subjects = [row[i].strip() for row in all_rows if row[i]]
        folder = os.path.join(input_dir, f"{month}M")

        for filename in os.listdir(folder):
            match = pattern.match(filename)
            if match:  # match the expected pattern
                subject_id = match.group(1)  # extract subject ID of the file
                # print(f"Extracted subject ID: {subject_id}")
                # print(f"Subjects in month {month}: {subjects[:5]}...")
                if subject_id in subjects:  # subject id is in the list of subjects for this month
                    # print(f"Found file for subject {subject_id} -> {filename}")
                    # month + 1 would indicate the month in which the recording is from
                    subject_files[subject_id].append((filename, month))
                else:
                    print(
                        f"File {filename} does not match any subject in month {month}. Skipping.")
            else:
                print(
                    f"File {filename} does not match the expected pattern. Skipping.")


# # Step 2: Split subject IDs
all_subjects = list(subject_files.keys())
random.seed(42)
random.shuffle(all_subjects)

n_total = len(all_subjects)
n_train = int(n_total * 0.8)


train_subjects = set(all_subjects[:n_train])
test_subjects = set(all_subjects[n_train:])

splits = {
    'train': train_subjects,
    'test': test_subjects,
}

# src_path: /scratch/chntzi001/khula/3M/1_191_76009725_3_20221012_030501002_processed.set
# dst_path: /scratch/chntzi001/khula/train/1_191_76009725_3_20221012_030501002_processed.set

for split_name, subjects in splits.items():
    for subject in subjects:
        for filename, session in subject_files[subject]:
            input_dir = '/scratch/chntzi001/khula'
            input_dir = os.path.join(input_dir, f"{session}M")
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_root, split_name)
            if split_name == 'train':
                with open("trainLinReg.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")
            elif split_name == 'test':
                with open("testLinReg.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")


print("✅ Done: raw data organized by split and session.")
