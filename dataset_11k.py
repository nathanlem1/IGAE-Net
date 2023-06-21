"""
To learn more how to write custom datasets in PyTorch, look at the following links:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://learnopencv.com/multi-label-image-classification-with-pytorch/

Custom data loading for 11k hands dataset: https://sites.google.com/view/11khands

"""

import csv
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_age_category(age):
    """ Determines age groups
    Args:
        age (int): age of an individual

    Age range: 18 - 75. This age range is grouped into 6 as follows:
    group 1: <= 20
    group 2: 21
    group 3: 22
    group 4: 23
    group 5: >= 24 & <= 30
    group 6: > 30

    In this case, groups 21, 22 and 23 are kept as they are.
    """
    if int(age) <= 20:
        age = '0-20'
    elif (int(age) >= 24) and (int(age) <= 30):
        age = '24-30'
    elif int(age) > 30:
        age = '31-75'

    return age


class AttributesDataset:
    """
    Gets important attributes from the 11k hands dataset: https://sites.google.com/view/11khands
    """
    def __init__(self, annotation_path):
        """
        Args:
            annotation_path: path to the data annotation CSV file.
        """
        identity_labels = []
        gender_labels = []
        age_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                identity_labels.append(row['id'])
                gender_labels.append(row['gender'])
                age_labels.append(get_age_category(row['age']))

        self.identity_labels = np.unique(identity_labels)  # Extract all the unique labels for identity classes
        self.gender_labels = np.unique(gender_labels)
        self.age_labels = np.unique(age_labels)

        self.num_identities = len(self.identity_labels)
        self.num_genders = len(self.gender_labels)
        self.num_ages = len(self.age_labels)

        self.identity_id_to_name = dict(zip(range(len(self.identity_labels)), self.identity_labels))
        self.identity_name_to_id = dict(zip(self.identity_labels, range(len(self.identity_labels))))

        self.gender_id_to_name = dict(zip(range(len(self.gender_labels)), self.gender_labels))
        self.gender_name_to_id = dict(zip(self.gender_labels, range(len(self.gender_labels))))

        self.age_id_to_name = dict(zip(range(len(self.age_labels)), self.age_labels))
        self.age_name_to_id = dict(zip(self.age_labels, range(len(self.age_labels))))


class HandsDataset(Dataset):
    """
    Custom HandsDataset class which inherits from the PyTorch Dataset for 11k hands dataset loading.
    """
    def __init__(self, annotation_path, attributes, transform=None):
        """
        Args:
            annotation_path: path to the data annotation CSV file.
            attributes: attributes of the dataset.
            transform: optional transform to be applied on a sample for data augmentation.
        """
        super(HandsDataset).__init__()
        self.transform = transform
        self.attr = attributes

        # Initialize the arrays to store the ground truth labels and paths to the images.
        self.data = []
        self.identity_labels = []
        self.gender_labels = []
        self.age_labels = []

        # Read the annotations from the CSV file in __init__ but leave the reading of images to __getitem__. This is
        # memory efficient because all the images are not stored in the memory at once but read as required.
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.identity_labels.append(self.attr.identity_name_to_id[row['id']])
                self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
                self.age_labels.append(self.attr.age_name_to_id[get_age_category(row['age'])])

    def __len__(self):
        # Returns the total number of samples in the data
        return len(self.data)

    def __getitem__(self, idx):
        # Take the data sample (example) by its index idx.
        img_path = self.data[idx]

        # Read image
        img = Image.open(img_path)

        # Apply the data augmentations if needed
        if self.transform:
            img = self.transform(img)

        # Return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'identity_labels': self.identity_labels[idx],
                'gender_labels': self.gender_labels[idx],
                'age_labels': self.age_labels[idx]
            }
        }
        return dict_data


# Execute from the interpreter
if __name__ == "__main__":
    from torch.utils.data import DataLoader  # DataLoader is an iterator which provides features such as batching the
    # data, shuffling the data, loading the data in parallel using multiprocessing workers, etc.

    sub_dataset = 'dorsal_dr'
    attributes = AttributesDataset(os.path.join('./11k/sub_dataset', sub_dataset + '.csv'))

    # Investigate identity statistics of each sub-dataset - uncomment this to know it.
    from collections import Counter
    dataset = HandsDataset(os.path.join('./11k/sub_dataset', sub_dataset + '.csv'), attributes)
    count_id_dr = Counter(dataset.identity_labels)
    print('\nSub-data statistics: ')
    print('Number of identities in ' + sub_dataset + ': ', len(count_id_dr))
    print('Minimum number of samples per identity: ', min(list(count_id_dr.values())))
    print('Maximum number of samples per identity: ', max(list(count_id_dr.values())))
    print('Total number of samples: ', sum(list(count_id_dr.values())))

    # Train
    train_dataset = HandsDataset(os.path.join('./11k/trainval', sub_dataset, 'train.csv'), attributes)
    count_id_dr = Counter(train_dataset.identity_labels)
    print('\nSub-data statistics (train): ')
    print('Number of identities in ' + sub_dataset + ': ', len(count_id_dr))
    print('Minimum number of samples per identity: ', min(list(count_id_dr.values())))
    print('Maximum number of samples per identity: ', max(list(count_id_dr.values())))
    print('Total number of samples: ', sum(list(count_id_dr.values())))

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=8)

    # Val
    val_dataset = HandsDataset(os.path.join('./11k/trainval', sub_dataset, 'val.csv'), attributes)
    count_id_dr = Counter(val_dataset.identity_labels)
    print('\nSub-data statistics (val): ')
    print('Number of identities in ' + sub_dataset + ': ', len(count_id_dr))
    print('Minimum number of samples per identity: ', min(list(count_id_dr.values())))
    print('Maximum number of samples per identity: ', max(list(count_id_dr.values())))
    print('Total number of samples: ', sum(list(count_id_dr.values())))

    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True, num_workers=8)

    print('Done!')
