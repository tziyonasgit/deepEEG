import os
import shutil
import random
import re
from collections import defaultdict
import csv

# folder paths
input_dir = '/scratch/chntzi001/khula'
output_root = '/scratch/chntzi001/khula/processedBinLogReg/'
bayley_csv = '/home/chntzi001/deepEEG/setUpKhula/bayley/khulasubsBayley.csv'


# SUBID to COG value mapping
cog_map = {}
with open(bayley_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        subid = row['SUBID'].strip()
        cog = row['COG'].strip()
        if subid and cog:
            cog_map[subid] = int(cog)
        else:
            print(f"ERROR for SUBID{subid} and/or COG score{cog}")

# check folders for subject file
subject_files = defaultdict(list)  # list of subject file names
pattern = re.compile(r'^\d+_\d+_(\d+)_\d+_.*_processed\.set$')

months = [3, 24]
folder = os.path.join(input_dir, "24M")
if not os.path.isdir(folder):
    print(f'Missing folder: {folder}')
else:
    for filename in os.listdir(folder):
        if not filename.endswith('_processed.set'):
            continue
        m = pattern.match(filename)
        if not m:
            print(f'Skipping non-matching filename: {filename}')
            continue

        subject_id = m.group(1)
        if subject_id in cog_map:  # keep only if we have a label
            label = cog_map[subject_id]
            # dictionary of SUBID: [filename, label]
            subject_files[subject_id].append((filename, label))
        else:
            pass  # no label for this subject, skip

# subject-level split
all_subjects = list(subject_files.keys())
random.seed(42)
random.shuffle(all_subjects)

n_total = len(all_subjects)
# n_total =  30 # For testing purposes, limit to 100 subjects
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

train_subjects = set(all_subjects[:n_train])
val_subjects = set(all_subjects[n_train:n_train + n_val])
test_subjects = set(all_subjects[n_train + n_val:])

splits = {
    'train': train_subjects,
    'val': val_subjects,
    'test': test_subjects,
}

for split_name, subjects in splits.items():
    for subject in subjects:
        for filename, session in subject_files[subject]:
            input_dir = '/scratch/chntzi001/khula/24M'
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_root, split_name)
            if split_name == 'train':
                with open("trainBayley.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")
            elif split_name == 'val':
                with open("valBayley.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")
            elif split_name == 'test':
                with open("testBayley.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")


print("✅ Done: raw data organized by split and session.")
