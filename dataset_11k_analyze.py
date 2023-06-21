"""
This code is used for analyzing 11k dataset.

"""

import csv
from collections import Counter
import os
import pandas
from shutil import copyfile

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# You only need to change this line to your dataset path
data_path = './11k'
ext = 'jpg'
if not os.path.isdir(data_path):
    print('Please change the 11k_data_path!')

train_path = data_path + '/Hands'   # 11k Hands data set has 190 identities.
if not os.path.isdir(train_path):
    print('Please check if train_path exists or is set correctly!')

# Read Hands info
hand_info = data_path + '/HandInfo.csv'
hand_info_file = open(hand_info, 'r')
reader = csv.DictReader(hand_info_file)

print('---------- 11k Hands dataset investigation has started ----------------')

# Collect gender information
gender_all = []
gender_dr = []
gender_dl = []
gender_pr = []
gender_pl = []

# Collect age information
age_all = []
age_dr = []
age_dl = []
age_pr = []
age_pl = []

for row in reader:
    id = row['id']
    accessories = row['accessories']
    imageName = row['imageName']
    aspectOfHand = row['aspectOfHand']
    src_path = train_path + '/' + imageName
    gender_all.append(row['gender'])
    age_all.append(int(row['age']))
    if int(accessories) == 0 and aspectOfHand == 'dorsal right':
        gender_dr.append(row['gender'])
        age_dr.append(int(row['age']))
    elif int(accessories) == 0 and aspectOfHand == 'dorsal left':
        gender_dl.append(row['gender'])
        age_dl.append(int(row['age']))
    elif int(accessories) == 0 and aspectOfHand == 'palmar right':
        gender_pr.append(row['gender'])
        age_pr.append(int(row['age']))
    elif int(accessories) == 0 and aspectOfHand == 'palmar left':
        gender_pl.append(row['gender'])
        age_pl.append(int(row['age']))


print(len(age_all))
print('Min: ', min(age_all))
print('Max: ', max(age_all))

# Sort the age lists
age_all.sort()
age_dr.sort()
age_dl.sort()
age_pr.sort()
age_pl.sort()


# # ---- Plot gender lists --------   # Uncomment this if you want to look at gender plots

# count_gender = Counter(gender_all)
# df_gender_all = pandas.DataFrame.from_dict(count_gender, orient='index')
# df_gender_all.plot(kind='bar', title='All genders')
# plt.xlabel('Gender')
# plt.ylabel('Number of images')

# count_gender_dr = Counter(gender_dr)
# df_gender_dr = pandas.DataFrame.from_dict(count_gender_dr, orient='index')
# df_gender_dr.plot(kind='bar', title='dorsal right')
# plt.xlabel('Gender')
# plt.ylabel('Number of images')
#
# count_gender_dl = Counter(gender_dl)
# df_gender_dl = pandas.DataFrame.from_dict(count_gender_dl, orient='index')
# df_gender_dl.plot(kind='bar', title='dorsal left')
# plt.xlabel('Gender')
# plt.ylabel('Number of images')
#
# count_gender_pr = Counter(gender_pr)
# df_gender_pr = pandas.DataFrame.from_dict(count_gender_pr, orient='index')
# df_gender_pr.plot(kind='bar', title='palmar right')
# plt.xlabel('Gender')
# plt.ylabel('Number of images')
#
# count_gender_pl = Counter(gender_pl)
# df_gender_pl = pandas.DataFrame.from_dict(count_gender_pl, orient='index')
# df_gender_pl.plot(kind='bar', title='palmar left')
# plt.xlabel('Gender')
# plt.ylabel('Number of images')
#
# plt.show()


#  ---- Plot age lists  --------

# count_age = Counter(age_all)    # All ages
# df_all = pandas.DataFrame.from_dict(count_age, orient='index')
# df_all.plot(kind='bar', title='All ages')
# plt.xlabel('Age')
# plt.ylabel('Number of images')

count_age_dr = Counter(age_dr)  # Right dorsal
df_age_dr = pandas.DataFrame.from_dict(count_age_dr, orient='index')
df_age_dr.plot(kind='bar', title='Age distribution for right dorsal', legend=None)
plt.xlabel('Age')
plt.ylabel('Number of images')
plt.savefig('./checkpoints/dorsal_right_age.png')

# count_age_dl = Counter(age_dl)  # Left dorsal
# df_age_dl = pandas.DataFrame.from_dict(count_age_dl, orient='index')
# df_age_dl.plot(kind='bar', title='Left dorsal', legend=None)
# plt.xlabel('Age')
# plt.ylabel('Number of images')
#
# count_age_pr = Counter(age_pr)  # Right palmar
# df_age_pr = pandas.DataFrame.from_dict(count_age_pr, orient='index')
# df_age_pr.plot(kind='bar', title='Right palmar', legend=None)
# plt.xlabel('Age')
# plt.ylabel('Number of images')
#
# count_age_pl = Counter(age_pl)  # Left palmar
# df_age_pl = pandas.DataFrame.from_dict(count_age_pl, orient='index')
# df_age_pl.plot(kind='bar', title='Left palmar', legend=None)
# plt.xlabel('Age')
# plt.ylabel('Number of images')


# Determine age groups
"""
Age range: 18 - 75. This age range is grouped into 6 as follows:
group 1: <= 20
group 2: 21
group 3: 22
group 4: 23
group 5: >= 24 & <= 30
group 6: > 30
"""


def create_group(counter):
    g = Counter()
    g['0-20'] = 0
    g['21'] = 0
    g['22'] = 0
    g['23'] = 0
    g['24-30'] = 0
    g['31-75'] = 0

    for k, v in counter.items():
        if k <= 20:
            g['0-20'] += v   # group1
        elif k == 21:
            g['21'] += v   # group2
        elif k == 22:
            g['22'] += v   # group3
        elif k == 23:
            g['23'] += v   # group4
        elif (k >= 24) and (k <= 30):
            g['24-30'] += v    # group5
        elif k > 30:
            g['31-75'] += v   # group6

    return g


# Create age groups
g_dr = create_group(count_age_dr)  # Right dorsal
df_g_dr = pandas.DataFrame.from_dict(g_dr, orient='index')
df_g_dr.plot(kind='bar', title='Age group distribution for right dorsal', legend=None)
plt.xlabel('Age group')
plt.ylabel('Number of images')
plt.savefig('./checkpoints/dorsal_right_age_group.png')

# g_dl = create_group(count_age_dl)  # Left dorsal
# df_g_dl = pandas.DataFrame.from_dict(g_dl, orient='index')
# df_g_dl.plot(kind='bar', title='Left dorsal', legend=None)
# plt.xlabel('Age group')
# plt.ylabel('Number of images')
#
# g_pr = create_group(count_age_pr)  # Right palmar
# df_g_pr = pandas.DataFrame.from_dict(g_pr, orient='index')
# df_g_pr.plot(kind='bar', title='Right palmar', legend=None)
# plt.xlabel('Age group')
# plt.ylabel('Number of images')
#
# g_pl = create_group(count_age_pl)  # Left palmar
# df_g_pl = pandas.DataFrame.from_dict(g_pl, orient='index')
# df_g_pl.plot(kind='bar', title='Left palmar', legend=None)
# plt.xlabel('Age group')
# plt.ylabel('Number of images')


plt.show()


print('ok')
