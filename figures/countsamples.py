from collections import Counter
import re

# Path to your file
train_file = "/Users/cccohen/Downloads/train.txt"

# Extract patient IDs
patient_ids = []
with open(train_file, "r") as f:
    for line in f:
        line = line.strip()
        # regex to capture patient id (the 3rd number in the filename)
        match = re.match(r"^\d+_\d+_(\d+)_", line)
        if match:
            patient_ids.append(match.group(1))

# Count samples per patient
counts = Counter(patient_ids)

# Convert to array (list of counts)
counts_array = list(counts.values())

print("Counts dictionary:", counts)
print("Counts array:", counts_array)
