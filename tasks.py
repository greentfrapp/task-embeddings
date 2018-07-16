from keras.datasets import mnist
import numpy as np

class GeneralTask():

	def __init__(self):
		(self.x_train, self.y_train), _ = mnist.load_data()
		self.train_idx = np.where(self.y_train < 6)[0]
		np.random.shuffle(self.train_idx)
		self.idx = 0

	def next(self, batchsize):
		start = self.idx
		end = self.idx + batchsize
		if end > len(self.train_idx):
			minibatch = self.train_idx[start:]
			np.random.shuffle(self.train_idx)
			minibatch = np.concatenate(minibatch, self.train_idx[:(end - len(self.train_idx))], axis=0)
		else:
			minibatch = self.train_idx[start:end]
		return self.x_train[minibatch], self.y_train[minibatch]

class FewShotTask():

	def __init__(self):
		(self.x_train, self.y_train), _ = mnist.load_data()
		self.train_idx = {}
		for i in np.arange(6, 10):
			self.train_idx[i] = np.where(self.y_train == i)[0]
			np.random.shuffle(self.train_idx[i])
		self.idx = 0

	def next(self, batchsize, base=None):
		start = self.idx
		end = self.idx + batchsize

		if base is None:
			pool = self.train_idx[np.random.choice(np.arange(6, 10))]
		else:
			pool = self.train_idx[base]

		if end > len(pool):
			minibatch = pool[start:]
			np.random.shuffle(pool)
			minibatch = np.concatenate(minibatch, pool[:(end - len(pool))], axis=0)
		else:
			minibatch = pool[start:end]
		return self.x_train[minibatch], self.y_train[minibatch]

class FewShotTask2():

	def __init__(self):
		(self.x_train, self.y_train), _ = mnist.load_data()
		self.train_idx = {}
		for i in np.arange(10):
			self.train_idx[i] = np.where(self.y_train == i)[0]
			np.random.shuffle(self.train_idx[i])
		self.idx = 0
		self.rand_train_x = np.random.RandomState(1)
		self.rand_train_y = np.random.RandomState(1)
		self.rand_test_x = np.random.RandomState(2)
		self.rand_test_y = np.random.RandomState(2)

	def next(self):
		chosen_classes = np.random.choice(np.arange(10), size=3, replace=False)
		train_x = []
		train_y = []
		test_x = []
		test_y = []
		for i, label in enumerate(chosen_classes):
			train_x.append(self.x_train[np.random.choice(self.train_idx[label])])
			train_y.append(i)
		for i, label in enumerate(chosen_classes):
			test_x.append(self.x_train[np.random.choice(self.train_idx[label])])
			test_y.append(i)
		self.rand_train_x.shuffle(train_x)
		self.rand_train_y.shuffle(train_y)
		self.rand_test_x.shuffle(test_x)
		self.rand_test_y.shuffle(test_y)
		return train_x, train_y, test_x, test_y