# each sub will have all their .edf files concatenated into a single .edf
# edf files will be organised alphabetically from sub 1 ... sub 300
# there will be a csv file that will have the 5 scores of each sub and their combined score
# If their overall Bayley score is < N then will be considered abnormal else normal
# from here i need to think...


import csv
import os
import shutil
from pathlib import Path
import mne


def createFolder(folderName):
    try:
        os.mkdir(folderName)
        print(f"Directory '{folderName}' created successfully.")
    except FileExistsError:
        print(f"Directory '{folderName}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{folderName}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def splitData():
    createFolder("abnormal")
    createFolder("normal")

    ab = 0
    norm = 0

    srcRoot = "/Users/cccohen/quero"
    labelFile = "queroLabels.txt"

    with open(labelFile, "w") as f:

        with open('Quero_Subject_IDs.csv', 'r', newline='') as file:
            csvReader = csv.reader(file)
            header = next(csvReader)
            for row in csvReader:
                id = row[0]
                srcPath = os.path.join(srcRoot, id)
                if int(row[6]) >= 25:
                    ab += 1
                    shutil.copytree(srcPath, os.path.join("abnormal", id))
                    f.write(f"{id}\t{1}\n")
                else:
                    norm += 1
                    shutil.copytree(srcPath, os.path.join("normal", id))
                    f.write(f"{id}\t{0}\n")

        print(f"Abnormal patients: {ab}\n")
        print(f"Normal patients: {norm}\n")


def searchForEDF(folderName):
    for folder in folderName.iterdir():  # loops through every file/folder in directory
        if folder.is_dir():  # checks if it is a folder
            for edfFile in folder.rglob("*.edf"):
                if edfFile.is_file():
                    destination = folder / edfFile.name
                    shutil.move(str(edfFile), str(destination))


def concatenateEDF(folderName):
    rawFiles = []
    for edfFile in folderName.glob("*.edf"):
        raw = mne.io.read_raw_edf(edfFile, preload=True, verbose=False)
        rawFiles.append(raw)

    if rawFiles:
        unifiedEDF = mne.concatenate_raws(rawFiles, preload=True)
        fileName = str(folderName) + "_raw.fif"
        unifiedEDF.save(fileName, overwrite=True)
    else:
        print("No .edf files found in the directory.")


def delete_subfolders(parent_dir):
    for folder in parent_dir.iterdir():
        if folder.is_dir():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")


if __name__ == "__main__":

    splitData()

    abnormalDir = Path("/Users/cccohen/deepEEG/abnormal")
    normalDir = Path("/Users/cccohen/deepEEG/normal")

    searchForEDF(abnormalDir)
    searchForEDF(normalDir)

    for folder in abnormalDir.iterdir():
        concatenateEDF(folder)
    for folder in normalDir.iterdir():
        concatenateEDF(folder)

    # Delete subfolders after processing
    delete_subfolders(abnormalDir)
    delete_subfolders(normalDir)

    # for folder in abnormalDir.iterdir():
    #     fileName = str(folder) + "raw.fif"
    #     shutil.copy2(folder, fileName)
    # for folder in normalDir.iterdir():
    #     fileName = str(folder) + "raw.fif"
    #     os.rename(folder, fileName)
