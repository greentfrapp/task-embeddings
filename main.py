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

from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from tasks import FewShotTask2

FLAGS = flags.FLAGS

flags.DEFINE_integer("steps", 1000, "Number of training steps")


class Net(object):

	def __init__(self, inputs, layers=3):
		with tf.variable_scope("net", reuse=False):
			running_output = inputs
			for i in np.arange(layers):
				conv = tf.layers.conv2d(
					inputs=running_output,
					filters=32,
					kernel_size=(3, 3),
					padding="same",
					activation=tf.nn.relu,
					name="conv_{}".format(i),
					# reuse=tf.AUTO_REUSE,
					reuse=False,
				)
				maxpool = tf.layers.max_pooling2d(
					inputs=conv,
					pool_size=(2, 2),
					strides=(1, 1),
				)
				running_output = maxpool
			self.output = running_output


class CNN(object):

	def __init__(self, name, layers, output_dim):
		super(CNN, self).__init__()
		self.name = name
		self.hidden = 128
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(layers, output_dim)

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

	def build_model(self, layers, output_dim):

		# task-specific training set
		self.train_inputs = tf.placeholder(
			shape=(None, 28, 28),
			dtype=tf.float32,
			name="train_inputs",
		)
		train_inputs = tf.reshape(self.train_inputs, [-1, 28, 28, 1])
		self.train_labels = tf.placeholder(
			shape=(3),
			dtype=tf.int32,
			name="train_labels",
		)
		self.train_onehot_labels = tf.one_hot(
			indices=self.train_labels,
			depth=3,
		)

		# task-specific test set
		self.test_inputs = tf.placeholder(
			shape=(None, 28, 28),
			dtype=tf.float32,
			name="test_inputs",
		)
		test_inputs = tf.reshape(self.test_inputs, [-1, 28, 28, 1])
		self.test_labels = tf.placeholder(
			shape=(None),
			dtype=tf.int64,
			name="test_labels",
		)
		self.test_onehot_labels = tf.one_hot(
			indices=self.test_labels,
			depth=3,
		)

		# Calculate class vectors
		#	Embed training samples
		net = Net(train_inputs)
		running_output = tf.reshape(net.output, [-1, (28 - layers) * (28 - layers) * 32])
		train_samples_embed = tf.layers.dense(
			inputs=running_output,
			units=self.hidden,
			activation=None,
			name="train_samples_embed",
		)
		# 	Embed training labels
		train_labels_embed = tf.layers.dense(
			inputs=self.train_onehot_labels,
			units=self.hidden,
			activation=None,
			name="train_labels_embed",
		)
		self.train_embed = train_embed = train_samples_embed + train_labels_embed
		for i in np.arange(3):
			train_embed, _ = self.multihead_attention(
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
			train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=0)

		class_query = tf.get_variable(
			name="class_query",
			shape=(3, self.hidden),
			dtype=tf.float32,
		)
		for i in np.arange(3):
			class_query, _ = self.multihead_attention(
				query=class_query,
				key=class_query,
				value=class_query,
			)
			class_query, _ = self.multihead_attention(
				query=class_query,
				key=train_embed,
				value=train_embed,
			)
			dense = tf.layers.dense(
				inputs=class_query,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="decoder_layer{}_dense1".format(i + 1)
			)
			class_query += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="decoder_layer{}_dense2".format(i + 1)
			)
			class_query = tf.contrib.layers.layer_norm(class_query, begin_norm_axis=0)
		class_vectors = tf.layers.dense(
			inputs=class_query,
			units=(28 - layers) * (28 - layers) * 32,
			name="class_vectors",
		)
		
		net2 = Net(test_inputs)
		running_output = tf.reshape(net2.output, [-1, (28 - layers) * (28 - layers) * 32])

		self.running_output = running_output / tf.norm(running_output, keepdims=True)

		# assume no bias
		output_weights = class_vectors

		self.scale = tf.get_variable(
			name="scale",
			shape=(1),
			dtype=tf.float32,
		)

		self.output_weights = output_weights / tf.norm(output_weights, axis=1, keepdims=True)

		self.output = self.scale * tf.matmul(self.running_output, tf.transpose(self.output_weights))

		self.logits = self.output
		self.softmax = tf.nn.softmax(self.logits)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_onehot_labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

		self.test_accuracy = tf.contrib.metrics.accuracy(labels=self.test_labels, predictions=tf.argmax(self.logits, axis=1))
		

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
		# output = tf.contrib.layers.layer_norm(output, begin_norm_axis=1)
		return output, attention_weights

def main(unused_args):
	sess = tf.Session()
	model = CNN("cnn", layers=3, output_dim=6)
	sess.run(tf.global_variables_initializer())
	task = FewShotTask2()
	for steps in np.arange(FLAGS.steps):
		train_x, train_y, test_x, test_y = task.next()
		feed_dict = {
			model.train_inputs: train_x,
			model.train_labels: train_y,
			model.test_inputs: test_x,
			model.test_labels: test_y,
		}
		loss, acc, _ = sess.run([model.loss, model.test_accuracy, model.optimize], feed_dict)
		if steps % 100 == 0:
			print(loss, acc)


if __name__ == "__main__":
	app.run(main)
