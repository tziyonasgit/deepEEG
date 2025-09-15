import os
import shutil
import random
import re
from collections import defaultdict
import csv

# folder paths
input_dir = '/scratch/chntzi001/khula'
output_root = '/scratch/chntzi001/khula/processedLinReg/'
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

months = [3, 6, 12, 24]
for month in months:
    folder = os.path.join(input_dir, f"{month}M")
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
                subject_files[subject_id].append((filename, label, month))
            else:
                pass  # no label for this subject, skip

# subject-level split
all_subjects = list(subject_files.keys())
random.seed(42)
random.shuffle(all_subjects)

n_total = len(all_subjects)
# n_total =  30 # For testing purposes, limit to 100 subjects
n_train = int(n_total * 0.8)

train_subjects = set(all_subjects[:n_train])
test_subjects = set(all_subjects[n_train:])

splits = {
    'train': train_subjects,
    'test': test_subjects,
}

for split_name, subjects in splits.items():
    for subject in subjects:
        for filename, session, month in subject_files[subject]:
            input_dir = f'/scratch/chntzi001/khula/{month}M'
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_root, split_name)
            if split_name == 'train':
                with open("trainLinReg.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")
            elif split_name == 'test':
                with open("testLinReg.txt", "a") as f:
                    f.writelines(src_path + "_" + dst_path + "\n")


print("✅ Done: raw data organized by split and session.")
