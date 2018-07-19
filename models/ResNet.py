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


class ResBlock(object):

	def __init__(self, inputs, n_filters, name, is_training, csn=None):
		super(ResBlock, self).__init__()
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(n_filters, is_training, csn)

	def build_model(self, n_filters, is_training=False, csn=None):
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_1",
			reuse=tf.AUTO_REUSE,
		)
		bn_1 = tf.contrib.layers.batch_norm(
			inputs=conv_1,
			scope="bn_1",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			# is_training=False,
		)
		conv_2 = tf.layers.conv2d(
			inputs=bn_1,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_2",
			reuse=tf.AUTO_REUSE,
		)
		bn_2 = tf.contrib.layers.batch_norm(
			inputs=conv_2,
			scope="bn_2",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			# is_training=False,
		)
		conv_3 = tf.layers.conv2d(
			inputs=bn_2,
			filters=n_filters,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="conv_3",
			reuse=tf.AUTO_REUSE,
		)
		bn_3 = tf.contrib.layers.batch_norm(
			inputs=conv_3,
			scope="bn_3",
			reuse=tf.AUTO_REUSE,
			# is_training=is_training,
			# is_training=False,
		)
		res_conv = tf.layers.conv2d(
			inputs=self.inputs,
			filters=n_filters,
			kernel_size=(1, 1),
			padding="same",
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=None,
			name="res_conv",
			reuse=tf.AUTO_REUSE,
		)
		max_pool = tf.layers.max_pooling2d(
			inputs=bn_3+res_conv,
			pool_size=(2, 2),
			strides=(1, 1),
		)
		# seems like the gradient should be added prior to the relu
		# if csn is not None:
		# 	max_pool += csn[self.name]
		self.outputs = tf.nn.relu(max_pool)
		# if csn is not None:
		# 	output += tf.nn.relu(csn[self.name])
		# self.outputs = tf.layers.dropout(
		# 	inputs=output,
		# 	rate=0.5,
		# 	training=is_training,
		# )

# MiniResNet comprising several ResBlocks
class MiniResNet(object):

	def __init__(self, inputs, name="miniresnet", is_training=True, csn=None):
		super(MiniResNet, self).__init__()
		self.name = name
		self.inputs = inputs
		self.is_training = is_training
		self.csn = csn
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			# variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, parent.name + '/' + self.name)
			# self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):
		self.resblock_1 = ResBlock(self.inputs, 64, name="resblock_1", is_training=self.is_training)
		output = self.resblock_1.outputs
		self.resblock_2 = ResBlock(output, 96, name="resblock_2", is_training=self.is_training)
		output = self.resblock_2.outputs
		self.resblock_3 = ResBlock(output, 128, name="resblock_3", is_training=self.is_training, csn=self.csn)
		output = self.resblock_3.outputs
		self.resblock_4 = ResBlock(output, 256, name="resblock_4", is_training=self.is_training, csn=self.csn)
		output = self.resblock_4.outputs
		output = tf.layers.conv2d(
			inputs=output,
			filters=1024,
			kernel_size=(1, 1),
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=tf.nn.relu,
			name="main_conv_1",
			reuse=tf.AUTO_REUSE,
		)
		output = tf.layers.average_pooling2d(
			inputs=output,
			pool_size=(6, 6),
			strides=(1, 1),
		)
		output = tf.layers.dropout(
			inputs=output,
			rate=0.5,
			training=self.is_training,
		)
		self.output = tf.layers.conv2d(
			inputs=output,
			filters=384,
			kernel_size=(1, 1),
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			activation=None,
			name="main_conv_2",
			reuse=tf.AUTO_REUSE,
		)
		# output = tf.reshape(output, [-1, 19 * 19 * 384])
		# output = tf.layers.dense(
		# 	inputs=output,
		# 	units=n,
		# 	kernel_initializer=tf.contrib.layers.xavier_initializer(),
		# 	activation=None,
		# 	name="main_logits",
		# 	reuse=tf.AUTO_REUSE,
		# )
		# self.logits = output

class ResNet(object):

	def __init__(self, name, n_way=5, layers=3, input_tensors=None):
		super(ResNet, self).__init__()
		self.name = name
		self.hidden = 1024
		self.n_way = n_way
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
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 84, 84, 3])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 84, 84, 3])
			if tf.shape(input_tensors['train_labels'])[-1] != self.n_way:
				self.train_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['train_labels'], axis=2), depth=num_classes), [-1, num_classes])
				self.test_labels = tf.reshape(tf.one_hot(tf.argmax(input_tensors['test_labels'], axis=2), depth=num_classes), [-1, num_classes])
			else:
				self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, num_classes])
				self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, num_classes])

		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="training_flag",
		)

		batchsize = tf.shape(input_tensors['train_inputs'])[0]

		# Calculate class vectors
		#	Embed training samples
		self.net = net = MiniResNet(self.train_inputs)
		# (64, 5, 20000)
		# train_running_output = tf.reshape(net.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		train_running_output = tf.reshape(net.output, [batchsize, -1, 75 * 75 * 384])
		# (64, 5, 128)
		train_embed = tf.layers.dense(
			inputs=train_running_output,
			units=self.hidden,
			activation=None,
			name="train_embed",
		)
		
		for i in np.arange(5):
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

		net2 = MiniResNet(self.test_inputs)
		# running_output = tf.reshape(net2.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		running_output = tf.reshape(net2.output, [batchsize, -1, 75 * 75 * 384])

		self.running_output = running_output / tf.norm(running_output, axis=-1, keep_dims=True)

		output_weights = tf.layers.dense(
			inputs=train_embed,
			# units=(28 - layers) * (28 - layers) * 64,
			units=75 * 75 * 384,
			activation=None,
		)
		# ((64, 5, 5).T * (64, 5, 20000)) -> (64, 5, 20000) / (64, 5, 1)
		train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.n_way])
		# output_weights = tf.matmul(train_labels, output_weights, transpose_a=True)
		output_weights = tf.matmul(train_labels, output_weights, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
		self.output_weights = output_weights / tf.norm(output_weights, axis=-1, keep_dims=True)
		
		self.scale = tf.Variable(
			initial_value=10.,
			name="scale",
			# shape=(1),
			dtype=tf.float32,
		)

		# (64, 5, 20000) * (64, 5, 20000).T
		self.output = tf.matmul(self.running_output, self.output_weights, transpose_b=True)
		self.output = self.output * self.scale
		self.output = tf.reshape(self.output, [-1, self.n_way])

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

	