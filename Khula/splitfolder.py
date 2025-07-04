import os

sourceFolder = '3M'
subfolder1 = '3M_eval'
subfolder2 = '3M_test'

# Create subfolders if they do not exist
os.makedirs(subfolder1, exist_ok=True)
os.makedirs(subfolder2, exist_ok=True)
