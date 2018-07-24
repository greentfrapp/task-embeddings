"""
Architecture for Few-Shot Regression (Sinusoid and Multimodal)
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel


class FeatureExtractor(object):

	def __init__(self, inputs):
		self.n_units = [40, 40]
		with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
			self.build_model()

	def build_model(self):
		running_output = inputs
		for i, units in enumerate(self.n_units):
			running_output = tf.layers.dense(
				inputs=running_output,
				units=units,
				activation=tf.nn.relu,
				name="dense_{}".format(i)
			)
		self.output = running_output # shape = (meta_batch_size, num_shot_train, 40)


class FFN(BaseModel):

	def __init__(self, name, num_train_samples=10, num_test_samples=10):
		super(FFN, self).__init__()
		self.name = name
		# Attention parameters
		self.attention_layers = 3
		self.hidden = 64
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(num_train_samples, num_test_samples)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self, num_train_samples=10, num_test_samples=10):

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
		# use amplitude to scale loss during training
		self.amp = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
			name="amplitude"
		)

		batchsize = tf.shape(self.train_inputs)[0]

		# Extract training features
		train_feature_extractor = FeatureExtractor(self.train_inputs)
		
		# Calculate final weights with attention
		with tf.variable_scope("attention"):
			train_embed = tf.layers.dense(
				# concat features and labels
				inputs=tf.concat([train_feature_extractor.output, self.train_labels], axis=-1),
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
					name="encoder_layer{}_dense0".format(i)
				)
				train_embed += tf.layers.dense(
					inputs=dense,
					units=self.hidden,
					activation=None,
					name="encoder_layer{}_dense1".format(i)
				)
				train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

			train_embed = tf.layers.dense(
				inputs=train_embed,
				units=40,
				activation=None,
			)
			final_weights = tf.reduce_mean(train_embed, axis=1, keep_dims=True)

		# Extract test features
		test_feature_extractor = FeatureExtractor(self.test_inputs)
		test_features = test_feature_extractor.output

		self.predictions = tf.matmul(test_features, final_weights, transpose_b=True)
		
		amp = tf.reshape(self.amp, [-1, 1, 1])
		self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels / amp, [-1]), predictions=tf.reshape(self.predictions / amp, [-1]))
		
		self.optimize = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)
