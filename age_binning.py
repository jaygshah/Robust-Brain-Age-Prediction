import pandas as pd
import numpy as np
import os
import nibabel
import shutil

from collections import Counter
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
data = pd.read_csv("masterdata.csv")
age_dict, age_dict_rounded = {}, {}

# Create a dictionary with patient IDs as keys and their ages as values
for index, row in data.iterrows():
    age_dict[row['patientid']] = float(row['age'])

# Round the ages and store them in a new dictionary
for pid in age_dict:
    age_dict_rounded[pid] = round(age_dict[pid])

# Prepare the data for train-test split
X, y = [], []
for patient_id in age_dict_rounded:
    X.append(patient_id)
    y.append(age_dict_rounded[patient_id])

X = np.array(X)
y = np.array(y)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.10)

print(len(X), len(X_train), len(X_val), len(X_test))

# Remove the existing data directory if it exists
if os.path.exists(f"./data"):
    shutil.rmtree(f"./data")
    print(f"removed an existing ./data directory")

# Create directories for test, validation, and training data
for x in ["test", "val", "train"]:
    if not os.path.exists(f"./data/HC/{x}"):
        os.makedirs(f"./data/HC/{x}")
    for z in set(y_train):
        if not os.path.exists(f"./data/HC/{x}/{z}"):
            os.makedirs(f"./data/HC/{x}/{z}")

def get_file_path(patient_id):
    # Implement this function to return the correct file path for the given patient ID
    return f"./path/to/data/{patient_id}.nii"

# Create symbolic links for test data
for x in X_test:
    file_path = get_file_path(x)
    name = file_path.split("/")[-1]
    img = nibabel.load(file_path)
    try: 
        data = img.get_fdata()
        os.symlink(file_path, f"./data/HC/test/{age_dict_rounded[x]}/{name}")
    except:
        print("Corrupted file")
print("Done with test")

# Create symbolic links for validation data
for x in X_val:
    file_path = get_file_path(x)
    name = file_path.split("/")[-1]
    img = nibabel.load(file_path)
    try: 
        data = img.get_fdata()
        os.symlink(file_path, f"./data/HC/val/{age_dict_rounded[x]}/{name}")
    except:
        print("Corrupted file")
print("Done with val")

# Create symbolic links for training data
for x in X_train:
    file_path = get_file_path(x)
    name = file_path.split("/")[-1]
    img = nibabel.load(file_path)
    try: 
        data = img.get_fdata()
        os.symlink(file_path, f"./data/HC/train/{age_dict_rounded[x]}/{name}")
    except:
        print("Corrupted file")
print("Done with train")