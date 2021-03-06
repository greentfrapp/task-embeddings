"""
Architecture for Few-Shot CIFAR
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel


class FeatureExtractor(object):

	def __init__(self, inputs, is_training):
		self.inputs = inputs
		self.is_training = is_training
		self.n_filters = [64, 64, 64, 64]
		self.dropout_rate = [None, None, 0.1, 0.3]
		with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
			self.build_model()

	def build_model(self):
		running_output = self.inputs
		prev_filters = 3
		for i, filters in enumerate(self.n_filters):
			conv_weights = tf.get_variable(
				name='conv_weights_{}'.format(i),
				shape=[3, 3, prev_filters, filters],
				initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				dtype=tf.float32,
			)
			conv_weights = tf.reshape(conv_weights, [9, prev_filters, filters])
			conv_weights = tf.nn.l2_normalize(conv_weights, axis=0)
			conv_weights = tf.reshape(conv_weights, [3, 3, prev_filters, filters])
			prev_filters = filters
			conv_bias = tf.Variable(
				initial_value=tf.zeros([filters]),
				name='conv_bias_{}'.format(i),
			)
			conv = tf.nn.conv2d(
				input=running_output,
				filter=conv_weights,
				strides=(1, 1, 1, 1),
				padding='SAME',
				name='conv_{}'.format(i),
			)
			normalizer = 27. / tf.cast(tf.shape(running_output)[1] * tf.shape(running_output)[2] * tf.shape(running_output)[3], tf.float32) * tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(running_output, axis=-1), axis=-1), axis=-1)
			normalizer = tf.reshape(normalizer, [-1, 1, 1, 1])
			conv = conv / normalizer
			conv = conv + conv_bias
			# conv = tf.layers.conv2d(
			# 	inputs=running_output,
			# 	filters=filters,
			# 	kernel_size=(3, 3),
			# 	strides=(1, 1),
			# 	padding="same",
			# 	activation=None,
			# 	kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			# 	name="conv_{}".format(i),
			# 	reuse=tf.AUTO_REUSE,
			# )
			norm = tf.contrib.layers.batch_norm(
				inputs=conv,
				activation_fn=None,
				reuse=tf.AUTO_REUSE,
				scope="model/extractor/norm_{}".format(i),
				# is_training=self.is_training, # should be True for both metatrain and metatest
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
			if self.dropout_rate[i] is None:
				dropout = relu
			else:
				dropout = tf.layers.dropout(
					inputs=relu,
					rate=self.dropout_rate[i],
					training=self.is_training,
				)
			running_output = dropout
		self.output = running_output # shape = (meta_batch_size*num_shot_train, 2, 2, 64)

class CNN_cifar(BaseModel):

	def __init__(self, name, num_classes, input_tensors=None):
		super(CNN_cifar, self).__init__()
		self.name = name
		self.num_classes = num_classes
		# Attention parameters
		self.attention_layers = 3
		self.hidden = 64
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(input_tensors)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)

	def build_model(self, input_tensors=None):

		self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 32, 32, 3])
		self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 32, 32, 3])
		self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, self.num_classes])
		self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, self.num_classes])

		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="training_flag",
		)

		batchsize = tf.shape(input_tensors['train_inputs'])[0]
		num_shot_train = tf.shape(input_tensors['train_inputs'])[1]
		num_shot_test = tf.shape(input_tensors['test_inputs'])[1]

		# Add noise to train_inputs
		# noise_mask = tf.random_normal(
		# 	shape=(batchsize*num_shot_train, 32, 32, 3),
		# 	mean=1.0,
		# 	stddev=1.0,
		# 	dtype=tf.float32,
		# 	seed=None,
		# 	name="noise",
		# )

		# Extract training features
		train_feature_extractor = FeatureExtractor(self.train_inputs, self.is_training)
		train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.num_classes])
		train_features = tf.reshape(train_feature_extractor.output, [batchsize, -1, 2*2*64])
		# train_features /= tf.norm(train_features, axis=-1, keep_dims=True)
		self.train_features = train_features
		train_features = tf.nn.l2_normalize(train_features, dim=-1)
		# Take mean of features for each class
		class_weights = tf.matmul(train_labels, train_features, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
		
		# for i in range(5):
		# 	train_logits = tf.matmul(train_features, class_weights, transpose_b=True)
		# 	train_logits = tf.reshape(train_logits, [-1, self.num_classes])
		# 	train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=train_logits))
		# 	gradient = tf.gradients(train_loss, class_weights)
		# 	class_weights = class_weights - 0.01 * gradient[0]
		
		# # Calculate class weights with attention
		# with tf.variable_scope("attention"):
		# 	train_embed = tf.layers.dense(
		# 		inputs=class_weights,
		# 		units=self.hidden,
		# 		activation=None,
		# 		name="train_embed",
		# 	)
		# 	for i in np.arange(self.attention_layers):
		# 		train_embed, _ = self.attention(
		# 			query=train_embed,
		# 			key=train_embed,
		# 			value=train_embed,
		# 		)
		# 		dense = tf.layers.dense(
		# 			inputs=train_embed,
		# 			units=self.hidden * 2,
		# 			activation=tf.nn.relu,
		# 			kernel_initializer=tf.contrib.layers.xavier_initializer(),
		# 			name="attention_layer{}_dense0".format(i),
		# 		)
		# 		train_embed += tf.layers.dense(
		# 			inputs=dense,
		# 			units=self.hidden,
		# 			activation=None,
		# 			kernel_initializer=tf.contrib.layers.xavier_initializer(),
		# 			name="attention_layer{}_dense1".format(i)
		# 		)
		# 		train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

		# 	class_weights = tf.layers.dense(
		# 		inputs=train_embed,
		# 		units=2*2*64,
		# 		activation=None,
		# 		kernel_initializer=tf.contrib.layers.xavier_initializer(),
		# 	)

		# class_weights = tf.get_variable(
		# 	name="init_weights",
		# 	shape=(1, 5, 2*2*64),
		# 	dtype=tf.float32,
		# 	trainable=True,
		# )
		# class_weights = tf.tile(class_weights, multiples=[batchsize, 1, 1])

		# Gradient descent on training set
		# for i in np.arange(5):
		# 	train_logits = tf.matmul(train_features, class_weights, transpose_b=True)
		# 	train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=train_logits))
		# 	grad = tf.gradients(train_loss, class_weights)[0]
		# 	class_weights = class_weights - 0.01 * grad

		# Extract test features
		test_feature_extractor = FeatureExtractor(self.test_inputs, self.is_training)
		test_features = tf.reshape(test_feature_extractor.output, [batchsize, -1, 2*2*64])

		class_weights = tf.nn.l2_normalize(class_weights, dim=-1)
		test_features = tf.nn.l2_normalize(test_features, dim=-1)
		
		# class_weights /= tf.norm(class_weights, axis=-1, keep_dims=True)
		# test_features /= tf.norm(test_features, axis=-1, keep_dims=True)

		self.scale = tf.Variable(
			initial_value=10.,
			name="scale",
			# shape=(1),
			dtype=tf.float32,
		)

		logits = tf.matmul(test_features, class_weights, transpose_b=True)
		logits = logits * self.scale
		self.logits = logits = tf.reshape(logits, [-1, self.num_classes])

		# Regularize with GOR loss https://arxiv.org/abs/1708.06320
		# Use training or test samples?
		#	Calculate 1st moment
		# moment_1 = tf.matmul(output_weights, output_weights, transpose_b=True)
		# moment_1 = moment_1 - tf.matrix_band_part(moment_1, -1, 0)
		#	Calculate 2nd moment
		# moment_2 = (moment_1 ** 2)
		#	Regularization Loss
		# n_pairs = self.num_classes * (self.num_classes - 1) / 2
		# moment_1 = tf.reduce_sum(moment_1) / tf.cast(n_pairs, dtype=tf.float32)
		# moment_2 = tf.reduce_sum(moment_2) / tf.cast(n_pairs, dtype=tf.float32)
		# loss_gor = (moment_1 ** 2) + tf.maximum(0., moment_2 - 1 / (2 * 2 * 64))

		# L2 Regularization for weights
		loss_l2 = tf.reduce_mean(tf.nn.l2_loss(class_weights))

		# regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/attention')])
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss + 0.1 * loss_l2)
		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=tf.argmax(self.logits, axis=1))
