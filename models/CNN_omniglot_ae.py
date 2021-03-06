"""
Architecture for Few-Shot Omniglot
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel


class FeatureExtractor(object):

	def __init__(self, inputs, is_training):
		self.inputs = inputs
		self.is_training = is_training
		self.n_filters = [64, 64, 64, 64]
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
				activation_fn=tf.nn.relu,
				reuse=tf.AUTO_REUSE,
				scope="model/net/norm_{}".format(i),
				# is_training=self.is_training, # should be True for both metatrain and metatest
			)
			maxpool = tf.layers.max_pooling2d(
				inputs=norm,
				pool_size=(2, 2),
				strides=(2, 2),
				padding="valid",
			)
			running_output = maxpool
		self.output = running_output # shape = (meta_batch_size*num_shot_train, 1, 1, 64)

class Decoder(object):

	def __init__(self, inputs):
		self.inputs = inputs
		self.n_units = [1000, 1000, 1000, 784]
		with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
			self.build_model()

	def build_model(self):
		running_output = self.inputs
		for i, units in enumerate(self.n_units):
			running_output = tf.layers.dense(
				inputs=running_output,
				units=units,
				name='dense_{}'.format(i),
			)
		running_output = tf.reshape(running_output, [-1, 28, 28, 1])
		self.output = running_output # shape = (meta_batch_size*num_shot_train, 1, 1, 64)


class CNN_omniglot(BaseModel):

	def __init__(self, name, num_classes=5, input_tensors=None):
		super(CNN_omniglot, self).__init__()
		self.name = name
		self.num_classes = num_classes
		# Attention parameters
		self.attention_layers = 3
		self.hidden = 64
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(input_tensors)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self, input_tensors=None):

		self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1])
		self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 28, 28, 1])
		self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, self.num_classes])
		self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, self.num_classes])

		self.is_training = tf.placeholder(
			shape=(None),
			dtype=tf.bool,
			name="training_flag",
		)

		batchsize = tf.shape(input_tensors['train_inputs'])[0]

		# Extract training features
		train_feature_extractor = FeatureExtractor(self.train_inputs, self.is_training)
		train_features = tf.reshape(train_feature_extractor.output, [batchsize, -1, 64])

		# Autoencoder
		decoder = Decoder(tf.reshape(train_features, [-1, 64]))
		reconstruction = decoder.output
		ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/extractor') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/decoder')
		ae_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.train_inputs, predictions=decoder.output))
		self.ae_optimize = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(ae_loss, var_list=ae_vars)

		# train_features /= tf.norm(train_features, axis=-1, keep_dims=True)
		train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.num_classes])
		self.train_features = train_features
		# Take mean of features for each class
		output_weights = tf.matmul(train_labels, train_features, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
		
		# Calculate class weights with attention
		with tf.variable_scope("attention"):
			train_embed = tf.layers.dense(
				inputs=output_weights,
				units=self.hidden,
				activation=None,
				name="train_embed",
			)
			for i in np.arange(self.attention_layers):
				train_embed, _ = self.attention(
					query=train_embed,
					key=train_embed,
					value=train_embed,
				)
				dense = tf.layers.dense(
					inputs=train_embed,
					units=self.hidden * 2,
					activation=tf.nn.relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name="encoder_layer{}_dense1".format(i + 1)
				)
				train_embed += tf.layers.dense(
					inputs=dense,
					units=self.hidden,
					activation=None,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name="encoder_layer{}_dense2".format(i + 1)
				)
				train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

			class_weights = tf.layers.dense(
				inputs=train_embed,
				units=64,
				activation=None,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
			)

		# Extract test features
		test_feature_extractor = FeatureExtractor(self.test_inputs, self.is_training)
		test_features = tf.reshape(test_feature_extractor.output, [batchsize, -1, 64])
		
		# class_weights /= tf.norm(class_weights, axis=-1, keep_dims=True)
		# test_features /= tf.norm(test_features, axis=-1, keep_dims=True)

		# self.scale = tf.Variable(
		# 	initial_value=10.,
		# 	name="scale",
		# 	# shape=(1),
		# 	dtype=tf.float32,
		# )

		logits = tf.matmul(test_features, class_weights, transpose_b=True)
		# logits = logits * self.scale
		self.logits = logits = tf.reshape(logits, [-1, self.num_classes])

		# regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/attention')])
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.logits))
		att_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/attention')
		self.optimize = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss, var_list=att_vars)
		self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=tf.argmax(self.logits, axis=1))
