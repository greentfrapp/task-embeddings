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
		self.n_filters = [96, 192, 384, 512]
		self.dropout_rate = [None, None, 0.1, 0.3]
		with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
			self.build_model()

	def build_model(self):
		running_output = self.inputs
		for i, filters in enumerate(self.n_filters):
			conv = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding="same",
				activation=None,
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name="conv_{}".format(i),
				reuse=tf.AUTO_REUSE,
			)
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
		self.output_dim = tf.shape(running_output)[1] * tf.shape(running_output)[2] * tf.shape(running_output)[3]

class CNN_cifar(BaseModel):

	def __init__(self, name, num_classes, input_tensors=None):
		super(CNN_cifar, self).__init__()
		self.name = name
		self.num_classes = num_classes
		# Attention parameters
		self.attention_layers = 5
		self.hidden = 64
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(input_tensors)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)

	def build_model(self, input_tensors=None):

		if input_tensors is None:
			self.m_train_inputs = tf.placeholder(
				shape=(None, 5*5, 32*32*3),
				dtype=tf.float32,
			)
			self.train_inputs = tf.reshape(self.m_train_inputs, [-1, 32, 32, 3])
			self.m_test_inputs = tf.placeholder(
				shape=(None, 5*1, 32*32*3),
				dtype=tf.float32,
			)
			self.test_inputs = tf.reshape(self.m_test_inputs, [-1, 32, 32, 3])
			self.m_train_labels = tf.placeholder(
				shape=(None, 5*5, 5),
				dtype=tf.float32,
			)
			self.train_labels = tf.reshape(self.m_train_labels, [-1, self.num_classes])
			self.m_test_labels = tf.placeholder(
				shape=(None, 5*1, 5),
				dtype=tf.float32,
			)
			self.test_labels = tf.reshape(self.m_test_labels, [-1, self.num_classes])
		
			batchsize = tf.shape(self.m_train_inputs)[0]
			num_shot_train = tf.shape(self.m_train_inputs)[1]
			num_shot_test = tf.shape(self.m_test_inputs)[1]

		else:
			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 32, 32, 3])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 32, 32, 3])
			self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, self.num_classes])
			self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, self.num_classes])

			batchsize = tf.shape(input_tensors['train_inputs'])[0]
			num_shot_train = tf.shape(input_tensors['train_inputs'])[1]
			num_shot_test = tf.shape(input_tensors['test_inputs'])[1]

		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="training_flag",
		)

		# Extract training features
		train_feature_extractor = FeatureExtractor(self.train_inputs, self.is_training)
		train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.num_classes])
		train_features = tf.reshape(train_feature_extractor.output, [batchsize, -1, train_feature_extractor.output_dim])
		train_features = tf.nn.l2_normalize(train_features, dim=-1)
		# train_features /= tf.norm(train_features, axis=-1, keep_dims=True)
		self.train_features = train_features
		# Take mean of features for each class
		output_weights = tf.matmul(train_labels, train_features, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
		# weights_transformer = tf.get_variable(
		# 	shape=(1, 2*2*512),
		# 	name='weights_transformer'
		# )
		# output_weights = output_weights * weights_transformer
		output_weights = tf.nn.l2_normalize(output_weights, dim=-1)

		# output_weights = tf.reshape(output_weights, [batchsize, -1, 2*2*64])

		# Calculate class weights with attention
		# with tf.variable_scope("attention"):
		# 	train_embed = tf.layers.dense(
		# 		inputs=output_weights,
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
		# 		dense = tf.layers.dropout(
		# 			inputs=dense,
		# 			rate=0.2,
		# 			training=self.is_training,
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

		# Extract test features
		test_feature_extractor = FeatureExtractor(self.test_inputs, self.is_training)
		test_features = tf.reshape(test_feature_extractor.output, [batchsize, -1, test_feature_extractor.output_dim])

		# class_weights = tf.nn.l2_normalize(class_weights, dim=-1)
		test_features = tf.nn.l2_normalize(test_features, dim=-1)
		
		# class_weights /= tf.norm(class_weights, axis=-1, keep_dims=True)
		# test_features /= tf.norm(test_features, axis=-1, keep_dims=True)

		self.scale = tf.Variable(
			initial_value=10.,
			name="scale",
			# shape=(1),
			dtype=tf.float32,
		)

		logits = tf.matmul(test_features, output_weights, transpose_b=True)
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
		# loss_l2 = tf.reduce_mean(tf.nn.l2_loss(class_weights))

		# regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/attention')])
		# self.loss = tf.losses.mean_squared_error(labels=self.test_labels, predictions=self.logits)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=tf.argmax(self.logits, axis=1))
