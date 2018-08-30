"""
Utility functions
"""

import os
import random
import tensorflow as tf
import numpy as np


def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

## Image helper
## Copied from Finn's implementation https://github.com/cbfinn/maml/blob/master/utils.py
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def u(x, t, B_n=1., n=1., k=1., L=1.):
    return B_n * np.sin(n * np.pi * x / L) * np.exp(-k * (n * np.pi / L)**2 * t)

def f(x, L):
    return (x == L).astype(np.float32)

def point_source(L, point, t, k=50., T_1=0., T_2=1.):
    step = 1e-2
    if point != 0.:
        L_1 = point
        x_1 = np.arange(0, L_1 + step * 2, step)
        t_1 = np.ones_like(x_1) * t
        components = []
        for n in np.arange(1, 10):
            B_n = (2. / L_1) * np.trapz(f(x_1, L_1) - (T_1 + ((T_2 - T_1) / L_1) * x_1) * np.sin(n * np.pi * x_1 / L_1), x_1)
            components.append(u(x_1, t_1, B_n=B_n, n=n, k=k, L=L_1))
        y_1 = np.sum(components, axis=0) + T_1 + ((T_2 - T_1) / L_1) * x_1

    if point != 5.:
        L_2 = L - point
        x_2 = np.arange(0, L_2 + step * 2, step)
        t_2 = np.ones_like(x_2) * t
        components = []
        for n in np.arange(1, 10):
            B_n = (2. / L_2) * np.trapz(f(x_2, L_2) - (T_1 + ((T_2 - T_1) / L_2) * x_2) * np.sin(n * np.pi * x_2 / L_2), x_2)
            components.append(u(x_2, t_2, B_n=B_n, n=n, k=k, L=L_2))
        y_2 = np.sum(components, axis=0) + T_1 + ((T_2 - T_1) / L_2) * x_2

    if point == 5.:
        return x_1, y_1
    if point == 0.:
        return x_2 + point, y_2[::-1]
    return np.concatenate([x_1, x_2 + point]), np.concatenate([y_1, y_2[::-1]])
