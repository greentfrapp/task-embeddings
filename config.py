"""
Stores default and min/max configuration for different datasources
"""

import numpy as np
import json as json


def save_config(hparams, folder, filename=None):
	filename = filename or 'config.json'
	with open(folder + filename, 'w') as file:
		json.dump(hparams, file)

def load_config(filename, test=False, notes=None):
	with open(filename, 'r') as file:
		hparams = json.load(file)
	if test:
		hparams['mode'] = 'test'
		hparams['notes'] = notes
	return hparams

def check_default_config(hparams):
	datasource = hparams['datasource']
	if datasource not in DEFAULT_CONFIGS:
		return hparams
	default_config = DEFAULT_CONFIGS[datasource]
	max_config = MAX_CONFIGS[datasource]
	for param_id, param_val in hparams.items():
		hparams[param_id] = param_val or default_config[param_id]
		# Check for values exceeding max
		if param_id in NUMERICAL_PARAMS:
			if hparams[param_id] > max_config[param_id]:
				print('Parameter {} exceeded maximum value of {}, clipping to maximum value...'.format(param_id, max_config[param_id]))
				hparams[param_id] = max_config[param_id]
	return hparams

NUMERICAL_PARAMS = [
	'num_classes_train',
	'num_classes_val',
	'num_classes_test',
	'num_shot_train',
	'num_shot_test',
	'steps',
	'meta_batch_size',
	'meta_lr',
]

DEFAULT_CONFIGS = {}
MAX_CONFIGS = {}

# CIFAR
DEFAULT_CONFIGS['omniglot'] = {
	'mode': None,
	'datasource': 'omniglot',
	'num_classes_train': 5,
	'num_classes_val': 5,
	'num_classes_test': 5,
	'num_shot_train': 1,
	'num_shot_test': 1,
	'steps': 40000,
	'meta_batch_size': 32,
	'meta_lr': 3e-4,
	'notes': None,
}
DEFAULT_CONFIGS['cifar'] = {
	'mode': None,
	'datasource': 'cifar',
	'num_classes_train': 5,
	'num_classes_val': 5,
	'num_classes_test': 5,
	'num_shot_train': 1,
	'num_shot_test': 1,
	'steps': 60000,
	'meta_batch_size': 4,
	'meta_lr': 1e-3,
	'notes': None,
}
DEFAULT_CONFIGS['miniimagenet'] = {
	'mode': None,
	'datasource': 'cifar',
	'num_classes_train': 5,
	'num_classes_val': 5,
	'num_classes_test': 5,
	'num_shot_train': 1,
	'num_shot_test': 1,
	'steps': 60000,
	'meta_batch_size': 4,
	'meta_lr': 1e-3,
	'notes': None,
}
MAX_CONFIGS['omniglot'] = {
	'mode': None,
	'datasource': 'omniglot',
	'num_classes_train': 1100,
	'num_classes_val': 100,
	'num_classes_test': 423,
	'num_shot_train': 599,
	'num_shot_test': 599,
	'steps': np.inf,
	'meta_batch_size': 200000,
	'meta_lr': np.inf,
	'notes': None,
}
MAX_CONFIGS['cifar'] = {
	'mode': None,
	'datasource': 'cifar',
	'num_classes_train': 64,
	'num_classes_val': 16,
	'num_classes_test': 20,
	'num_shot_train': 599,
	'num_shot_test': 599,
	'steps': np.inf,
	'meta_batch_size': 200000,
	'meta_lr': np.inf,
	'notes': None,
}
MAX_CONFIGS['miniimagenet'] = {
	'mode': None,
	'datasource': 'cifar',
	'num_classes_train': 64,
	'num_classes_val': 16,
	'num_classes_test': 20,
	'num_shot_train': 599,
	'num_shot_test': 599,
	'steps': np.inf,
	'meta_batch_size': 200000,
	'meta_lr': np.inf,
	'notes': None,
}