"""
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)

Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run this script from the miniImagenet directory:
    cd data/miniImagenet/
    python proc_images.py
"""

from __future__ import print_function
import csv
import glob
import os

from PIL import Image

path_to_images = 'images/'

# all_images = glob.glob(path_to_images + '*')

# Resize images
# for i, image_file in enumerate(all_images):
#     im = Image.open(image_file)
#     im = im.resize((84, 84), resample=Image.LANCZOS)
#     im.save(image_file)
#     if i % 500 == 0:
#         print(i)

names = []
with open('train.txt', 'r') as file:
    for line in file:
        names.append(line.rstrip('\n'))
names.sort()
for name in names:
    if not os.path.exists('train/' + name):
        print(name)
quit()

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    if not os.path.exists(datatype):
        os.system('mkdir ' + datatype)

    with open(datatype + '.txt', 'r') as file:
        for line in file:
            label = line.rstrip('\n')
            new_dir = datatype + '/'
            old_dir = path_to_images + label + '/'
            # os.system('mkdir ' + new_dir)
            # filenames = glob.glob(path_to_images + label + '/')
            # for old_dir in filenames:
            os.system('mv ' + old_dir + ' ' + new_dir)
