import argparse
import os
import shutil

import nibabel
import pandas as pd
from sklearn.model_selection import train_test_split


def get_file_path(patient_id):
    # masterdata.csv stores each scan's path in the patientid column; adapt this if yours stores IDs
    return patient_id


def link_split(split, patient_ids, ages, out_dir):
    linked = 0
    for patient_id, age in zip(patient_ids, ages):
        file_path = os.path.abspath(get_file_path(patient_id))
        try:
            nibabel.load(file_path).get_fdata()
        except Exception as e:
            print(f"Skipping unreadable or missing file {file_path}: {e}")
            continue

        age_dir = os.path.join(out_dir, "HC", split, str(age))
        os.makedirs(age_dir, exist_ok=True)
        os.symlink(file_path, os.path.join(age_dir, os.path.basename(file_path)))
        linked += 1
    print(f"Done with {split}: linked {linked}/{len(patient_ids)} scans")


def main():
    parser = argparse.ArgumentParser(description='Split scans into train/val/test folders by rounded age')
    parser.add_argument('--csv', default='masterdata.csv', type=str,
                        help='CSV with patientid and age columns')
    parser.add_argument('--out_dir', default='./data', type=str)
    parser.add_argument('--test_size', default=0.10, type=float)
    parser.add_argument('--val_size', default=0.10, type=float,
                        help='fraction of the remaining (non-test) scans used for validation')
    parser.add_argument('--seed', default=3141, type=int)
    args = parser.parse_args()

    # One age per scan, rounded to the closest integer age class
    data = pd.read_csv(args.csv).drop_duplicates('patientid', keep='last')
    X = data['patientid'].to_numpy()
    y = data['age'].astype(float).round().astype(int).to_numpy()

    # Split the data into training, validation, and test sets, stratified by age
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=args.test_size,
                                                        random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                      test_size=args.val_size, random_state=args.seed)

    print(len(X), len(X_train), len(X_val), len(X_test))

    hc_dir = os.path.join(args.out_dir, "HC")
    if os.path.exists(hc_dir):
        shutil.rmtree(hc_dir)
        print(f"removed an existing {hc_dir} directory")

    link_split("test", X_test, y_test, args.out_dir)
    link_split("val", X_val, y_val, args.out_dir)
    link_split("train", X_train, y_train, args.out_dir)


if __name__ == '__main__':
    main()
