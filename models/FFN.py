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

	def __init__(self, inputs, layers=2):
		with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
			running_output = inputs
			for i in np.arange(layers):
				running_output = tf.layers.dense(
					inputs=running_output,
					units=40,
					activation=tf.nn.relu,
					name="dense_{}".format(i)
				)
			self.output = running_output


class FFN(object):

	def __init__(self, name, layers=2, num_train_samples=10, num_test_samples=10):
		super(FFN, self).__init__()
		self.name = name
		self.hidden = 64
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(layers, num_train_samples, num_test_samples)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self, layers=2, num_train_samples=10, num_test_samples=10):

		self.train_inputs = tf.placeholder(
			shape=(None, num_train_samples, 1),
			dtype=tf.float32,
			name="train_inputs",
		)
		self.train_labels = tf.placeholder(
			shape=(None, num_train_samples, 1),
			dtype=tf.float32,
			name="train_labels",
		)
		self.test_inputs = tf.placeholder(
			shape=(None, num_test_samples, 1),
			dtype=tf.float32,
			name="test_inputs"
		)
		self.test_labels = tf.placeholder(
			shape=(None, num_test_samples, 1),
			dtype=tf.float32,
			name="test_labels",
		)
		# # use amplitude to scale loss
		# self.amp = tf.placeholder(
		# 	shape=(None),
		# 	dtype=tf.float32,
		# 	name="amplitude"
		# )

		batchsize = tf.shape(self.train_inputs)[0]

		# Calculate class vectors
		#	Embed training samples
		# (64, 10, 40)
		self.net = net = Net(self.train_inputs, layers)
		# (64, 5, 128)
		train_embed = tf.layers.dense(
			inputs=tf.concat([net.output, self.train_labels], axis=-1),
			# inputs=net.output,
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
		# (64, 10, 40)
		self.running_output = net2.output

		output_weights = tf.layers.dense(
			inputs=train_embed,
			# units=(28 - layers) * (28 - layers) * 64,
			units=40,
			activation=None,
		)
		# (64, 1, 40)
		self.output_weights = tf.reduce_mean(output_weights, axis=1, keep_dims=True)
		
		self.output = tf.matmul(self.running_output, self.output_weights, transpose_b=True)
		self.output = tf.reshape(self.output, [-1])
		self.predictions = self.output
		
		self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels, [-1]), predictions=self.output)
		
		self.optimize = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)

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

	