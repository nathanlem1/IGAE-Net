"""
Divide the 11k hands dataset into the dorsal right, dorsal left, palmar right and palmar left sub-datasets. It also
divides each sub-dataset into train and val (test) sets.

"""

import argparse
import csv
import os
import numpy as np
from tqdm import tqdm


def save_csv(data, path, fieldnames=('image_path', 'id', 'gender', 'age')):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split the 11k Hands dataset.')
    parser.add_argument('--input', default='./11k',
                        type=str, help="Path to the dataset: './11k'")

    args = parser.parse_args()
    input_folder = args.input
    annotation = os.path.join(input_folder, 'HandInfo.csv')

    # Store each 11k hands sub-dataset
    all_data_dr = []
    all_data_dl = []
    all_data_pr = []
    all_data_pl = []

    print('-----Data split has started -------')

    # Open annotation file
    with open(annotation) as csv_file:
        # Parse it as CSV
        reader = csv.DictReader(csv_file)

        # tqdm shows pretty progress bar. Each row in the CSV file corresponds to the image
        for row in tqdm(reader, total=reader.line_num):
            # We use aspectOfHand and accessories to exclude any hand image with accessories
            accessories = row['accessories']
            aspectOfHand = row['aspectOfHand']
            # We need image name to build the path to the image file
            img_name = row['imageName']
            # We're going to use image id and two other attributes.
            identity = row['id']
            gender = row['gender']
            age = row['age']
            img_path = os.path.join(input_folder, 'Hands', img_name)
            if int(accessories) == 0 and aspectOfHand == 'dorsal right':
                if int(identity) == 1200000:  # To change '1200000 ' to '1200000'. One image has this deviation!
                    identity = '1200000'
                all_data_dr.append([img_path, identity, gender, age])
            elif int(accessories) == 0 and aspectOfHand == 'dorsal left':
                all_data_dl.append([img_path, identity, gender, age])
            elif int(accessories) == 0 and aspectOfHand == 'palmar right':
                all_data_pr.append([img_path, identity, gender, age])
            elif int(accessories) == 0 and aspectOfHand == 'palmar left':
                all_data_pl.append([img_path, identity, gender, age])

    # Set the seed of the random number generator, so we can reproduce the results later
    np.random.seed(42)
    # Construct a Numpy array from the list
    all_data_dr = np.asarray(all_data_dr)
    all_data_dl = np.asarray(all_data_dl)
    all_data_pr = np.asarray(all_data_pr)
    all_data_pl = np.asarray(all_data_pl)
    # Take samples in random order
    inds_dr = np.random.choice(len(all_data_dr), len(all_data_dr), replace=False)
    inds_dl = np.random.choice(len(all_data_dl), len(all_data_dl), replace=False)
    inds_pr = np.random.choice(len(all_data_pr), len(all_data_pr), replace=False)
    inds_pl = np.random.choice(len(all_data_pl), len(all_data_pl), replace=False)

    # Create output folder for each sub-dataset if it doesn't exist
    if not os.path.isdir(os.path.join(input_folder, 'sub_dataset')):
        os.mkdir(os.path.join(input_folder, 'sub_dataset'))

    # Save each sub-dataset of 11k hands dataset: dorsal right, dorsal left, palmar right, palmar left
    save_csv(all_data_dr[inds_dr], os.path.join(input_folder, 'sub_dataset', 'dorsal_dr.csv'))
    save_csv(all_data_dl[inds_dl], os.path.join(input_folder, 'sub_dataset', 'dorsal_dl.csv'))
    save_csv(all_data_pr[inds_pr], os.path.join(input_folder, 'sub_dataset', 'palmar_pr.csv'))
    save_csv(all_data_pl[inds_pl], os.path.join(input_folder, 'sub_dataset', 'palmar_pl.csv'))

    # Create trainval folder for each sub-dataset if it doesn't exist: dorsal right, dorsal left, palmar right and
    # palmar left
    if not os.path.isdir(os.path.join(input_folder, 'trainval')):
        os.mkdir(os.path.join(input_folder, 'trainval'))
    if not os.path.isdir(os.path.join(input_folder, 'trainval', 'dorsal_dr')):
        os.mkdir(os.path.join(input_folder, 'trainval', 'dorsal_dr'))
    if not os.path.isdir(os.path.join(input_folder, 'trainval', 'dorsal_dl')):
        os.mkdir(os.path.join(input_folder, 'trainval', 'dorsal_dl'))
    if not os.path.isdir(os.path.join(input_folder, 'trainval', 'palmar_pr')):
        os.mkdir(os.path.join(input_folder, 'trainval', 'palmar_pr'))
    if not os.path.isdir(os.path.join(input_folder, 'trainval', 'palmar_pl')):
        os.mkdir(os.path.join(input_folder, 'trainval', 'palmar_pl'))

    # Split the data into train/val and save them as csv files. Here the val dataset is also used as the test dataset
    # for the final result. 50% for training and 50% for validation (testing)
    save_csv(all_data_dr[inds_dr][:int(0.5*len(all_data_dr))], os.path.join(input_folder, 'trainval', 'dorsal_dr',
                                                                            'train.csv'))
    save_csv(all_data_dr[inds_dr][int(0.5*len(all_data_dr)):], os.path.join(input_folder, 'trainval', 'dorsal_dr',
                                                                            'val.csv'))
    save_csv(all_data_dl[inds_dl][:int(0.5*len(all_data_dl))], os.path.join(input_folder, 'trainval', 'dorsal_dl',
                                                                            'train.csv'))
    save_csv(all_data_dl[inds_dl][int(0.5*len(all_data_dl)):], os.path.join(input_folder, 'trainval', 'dorsal_dl',
                                                                            'val.csv'))
    save_csv(all_data_pr[inds_pr][:int(0.5*len(all_data_pr))], os.path.join(input_folder, 'trainval', 'palmar_pr',
                                                                            'train.csv'))
    save_csv(all_data_pr[inds_pr][int(0.5*len(all_data_pr)):], os.path.join(input_folder, 'trainval', 'palmar_pr',
                                                                            'val.csv'))
    save_csv(all_data_pl[inds_pl][:int(0.5*len(all_data_pl))], os.path.join(input_folder, 'trainval', 'palmar_pl',
                                                                            'train.csv'))
    save_csv(all_data_pl[inds_pl][int(0.5*len(all_data_pl)):], os.path.join(input_folder, 'trainval', 'palmar_pl',
                                                                            'val.csv'))

    print('Done!')
