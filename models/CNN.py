"""
For each task
- Calculate class vectors
- Use cosine similarity to classify

TODO:
- Code for batchsize=1
- Test on MNIST 
- Test on Omniglot
- Extend to variable batchsize
"""

import tensorflow as tf
import numpy as np


class Net(object):

	def __init__(self, inputs, layers=3):
		with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
			running_output = inputs
			for i in np.arange(layers):
				conv = tf.layers.conv2d(
					inputs=running_output,
					filters=64,
					kernel_size=(3, 3),
					strides=(1, 1),
					padding="same",
					# activation=tf.nn.relu,
					activation=None,
					name="conv_{}".format(i),
					reuse=tf.AUTO_REUSE,
				)
				norm = tf.contrib.layers.batch_norm(
					inputs=conv,
					activation_fn=tf.nn.relu,
					reuse=tf.AUTO_REUSE,
					scope="model/net",
				)
				maxpool = tf.layers.max_pooling2d(
					inputs=norm,
					pool_size=(2, 2),
					strides=(2, 2),
					padding="valid",
				)
				running_output = maxpool
			self.output = running_output


class CNN(object):

	def __init__(self, name, layers=3, input_tensors=None):
		super(CNN, self).__init__()
		self.name = name
		self.hidden = 64
		self.n_way = 5
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(layers, input_tensors)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self, layers=3, input_tensors=None):

		if input_tensors is None:
			self.train_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="train_inputs",
			)
			self.train_labels = tf.placeholder(
				shape=(None, num_classes),
				dtype=tf.float32,
				name="train_labels",
			)
			self.test_inputs = tf.placeholder(
				shape=(None, 28, 28, 1),
				dtype=tf.float32,
				name="test_inputs"
			)
			self.test_labels = tf.placeholder(
				shape=(None, num_classes),
				dtype=tf.float32,
				name="test_labels",
			)

		else:
			num_classes = self.n_way
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 28, 28, 1])
			if tf.shape(input_tensors['train_labels'])[-1] != self.n_way:
				self.train_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['train_labels'], axis=2), depth=num_classes), [-1, num_classes])
				self.test_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['test_labels'], axis=2), depth=num_classes), [-1, num_classes])
			else:
				self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, num_classes])
				self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, num_classes])

		batchsize = tf.shape(input_tensors['train_inputs'])[0]

		# Calculate class vectors
		#	Embed training samples
		self.net = net = Net(self.train_inputs, layers)
		# (64, 5, 20000)
		# train_running_output = tf.reshape(net.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		train_running_output = tf.reshape(net.output, [batchsize, -1, 64])
		# (64, 5, 128)
		train_embed = tf.layers.dense(
			inputs=train_running_output,
			units=self.hidden,
			activation=None,
			name="train_embed",
		)
		
		for i in np.arange(3):
			train_embed, _ = self.attention(
				query=train_embed,
				key=train_embed,
				value=train_embed,
			)
			dense = tf.layers.dense(
				inputs=train_embed,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="encoder_layer{}_dense1".format(i + 1)
			)
			train_embed += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="encoder_layer{}_dense2".format(i + 1)
			)
			train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

		net2 = Net(self.test_inputs, layers)
		# running_output = tf.reshape(net2.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		running_output = tf.reshape(net2.output, [batchsize, -1, 64])

		self.running_output = running_output / tf.norm(running_output, axis=-1, keep_dims=True)

		output_weights = tf.layers.dense(
			inputs=train_embed,
			# units=(28 - layers) * (28 - layers) * 64,
			units=64,
			activation=None,
		)
		# (64, 5, 5).T * (64, 5, 20000)
		self.train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.n_way])
		output_weights = tf.matmul(self.train_labels, output_weights, transpose_a=True)
		self.output_weights = output_weights / tf.norm(output_weights, axis=-1, keep_dims=True)
		
		self.scale = tf.Variable(
			initial_value=10.,
			name="scale",
			# shape=(1),
			dtype=tf.float32,
		)

		# (64, 5, 20000) * (64, 5, 20000).T
		self.output = self.scale * tf.matmul(self.running_output, self.output_weights, transpose_b=True)
		self.output = tf.reshape(self.output, [-1, 5])

		self.logits = self.output
		
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)

		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=tf.argmax(self.logits, axis=1))
		
	def attention(self, query, key, value):
		dotp = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[-1], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(dotp)
		weighted_sum = tf.matmul(attention_weights, value)
		output = weighted_sum + query
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights

	def multihead_attention(self, query, key, value, h=4):
		W_query = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_key = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_value = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_output = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)

		multi_query = tf.reshape(tf.matmul(query, W_query), [-1, h, int(self.hidden/h)])
		multi_key = tf.reshape(tf.matmul(key, W_key), [-1, h, int(self.hidden/h)])
		multi_value = tf.reshape(tf.matmul(value, W_value), [-1, h, int(self.hidden/h)])

		dotp = tf.matmul(multi_query, multi_key, transpose_b=True) / (tf.cast(tf.shape(multi_query)[-1], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(dotp)

		weighted_sum = tf.matmul(attention_weights, multi_value)
		weighted_sum = tf.concat(tf.unstack(weighted_sum, axis=1), axis=-1)
		
		multihead = tf.matmul(weighted_sum, W_output)
		output = multihead + query
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights

	def save(self, sess, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, sess, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))

	