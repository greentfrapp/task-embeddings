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

	def __init__(self, inputs, layers=3, is_training=None):
		self.is_training = is_training
		with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
			running_output = inputs
			n_filters = [64, 64, 64, 64]
			for i in np.arange(layers):
				filters = n_filters[i]
				conv = tf.layers.conv2d(
					inputs=running_output,
					filters=filters,
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
					activation_fn=None,
					reuse=tf.AUTO_REUSE,
					# scope="model/net"
					scope="model/net/norm_{}".format(i),
					# is_training=self.is_training,
				)
				maxpool = tf.layers.max_pooling2d(
					inputs=norm,
					pool_size=(2, 2),
					strides=(2, 2),
					padding="valid",
				)
				relu = tf.nn.leaky_relu(
					features=maxpool,
					alpha=0.1,
				)
				running_output = relu
				if i == layers - 1 or i == layers - 2:
					if i == layers - 1:
						rate = 0.3
					else:
						rate = 0.1
					running_output = tf.layers.dropout(
						inputs=running_output,
						rate=rate,
						training=self.is_training,
					)
			self.output = running_output


class CNN_cifar(object):

	def __init__(self, name, n_way=5, layers=3, input_tensors=None, noise=False):
		super(CNN_cifar, self).__init__()
		self.name = name
		self.hidden = 128
		self.n_way = n_way
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(layers, input_tensors, noise)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self, layers=3, input_tensors=None, noise=False):

		num_classes = self.n_way
		self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 32, 32, 3])
		self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 32, 32, 3])
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
		self.net = net = Net(self.train_inputs, layers, self.is_training)
		# (64, 5, 20000)
		# train_running_output = tf.reshape(net.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		train_running_output = tf.reshape(net.output, [batchsize, -1, 2 * 2 * 64])
		noise_mask = tf.random_normal(
			shape=(batchsize, 1, 2 * 2 * 64),
			mean=1.0,
			stddev=1.0,
			dtype=tf.float32,
			seed=None,
			name="noise",
		)
		if noise:
			train_running_output = train_running_output * noise_mask
		# (64, 5, 128)

		with tf.variable_scope("attention"):
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
				# if i == 4:
				# 	dense = tf.layers.dropout(
				# 		inputs=train_embed,
				# 		rate=0.3,
				# 		training=self.is_training,
				# 	)
				# if i == 3:
				# 	dense = tf.layers.dropout(
				# 		inputs=train_embed,
				# 		rate=0.1,
				# 		training=self.is_training,
				# 	)
				train_embed += tf.layers.dense(
					inputs=dense,
					units=self.hidden,
					activation=None,
					name="encoder_layer{}_dense2".format(i + 1)
				)
				# train_embed = tf.layers.dropout(
				# 	inputs=train_embed,
				# 	rate=0.2,
				# 	training=self.is_training,
				# )
				train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

			output_weights = tf.layers.dense(
				inputs=train_embed,
				# units=(28 - layers) * (28 - layers) * 64,
				units=2 * 2 * 64,
				activation=None,
			)

		net2 = Net(self.test_inputs, layers, self.is_training)
		# running_output = tf.reshape(net2.output, [batchsize, -1, (28 - layers) * (28 - layers) * 64])
		running_output = tf.reshape(net2.output, [batchsize, -1, 2 * 2 * 64])

		self.running_output = running_output #/ tf.norm(running_output, axis=-1, keep_dims=True)

		# ((64, 5, 5).T * (64, 5, 20000)) -> (64, 5, 20000) / (64, 5, 1)
		train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.n_way])
		# output_weights = tf.matmul(train_labels, output_weights, transpose_a=True)
		output_weights = tf.matmul(train_labels, output_weights, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
		self.output_weights = output_weights #/ tf.norm(output_weights, axis=-1, keep_dims=True)
		
		# self.scale = tf.Variable(
		# 	initial_value=10.,
		# 	name="scale",
		# 	# shape=(1),
		# 	dtype=tf.float32,
		# )

		# (64, 5, 20000) * (64, 5, 20000).T
		self.output = tf.matmul(self.running_output, self.output_weights, transpose_b=True)
		# self.output = self.output * self.scale
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

	