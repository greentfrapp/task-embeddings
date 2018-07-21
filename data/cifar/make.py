import pickle
import numpy as np
from PIL import Image
import os

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

datafiles = ['test', 'train']

labels = [label.decode('utf-8') for label in unpickle('cifar-100-python/meta')[b'fine_label_names']]

if not os.path.exists('images/'):
	os.makedirs('images/')

for label in labels:
	if not os.path.exists('images/' + label):
		os.makedirs('images/' + label + '/')

for datafile in datafiles:
	data_dict = unpickle('cifar-100-python/' + datafile)
	for i, image in enumerate(data_dict[b'data']):
		label = labels[int(data_dict[b'fine_labels'][i])]
		new_path = 'images/' + label + '/' + data_dict[b'filenames'][i].decode('utf-8')
		Image.fromarray(np.transpose(image.reshape(3, 32, 32), axes=[1,2,0])).save(new_path)
		if i >0 and i % 500 == 0:
			print("Processed {} images...".format(i))