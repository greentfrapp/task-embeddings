from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

from time import strftime
import os
import json
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from absl import app

from models import CNN, CNN2, CNN_miniimagenet, FFN, ResNet, CNN_cifar, CNN_omniglot
from data_generator import DataGenerator
from config import check_default_config, save_config, load_config


FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool('train', False, 'Train')
flags.DEFINE_bool('test', False, 'Test')
flags.DEFINE_bool('load', False, 'Resume training from a saved model')
flags.DEFINE_bool('plot', False, 'Plot activations of training samples')
flags.DEFINE_bool('comet', False, 'Use comet.ml for logging')

# Task parameters
flags.DEFINE_string('datasource', 'omniglot', 'omniglot or sinusoid (miniimagenet WIP)')
flags.DEFINE_integer('num_classes', 5, 'Number of classes per task eg. 5-way refers to 5 classes')
flags.DEFINE_integer('num_shot_train', None, 'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('num_shot_test', None, 'Number of test samples per class per task')

# Training parameters
flags.DEFINE_integer('steps', None, 'Number of metatraining iterations')
flags.DEFINE_integer('meta_batch_size', 32, 'Batchsize for metatraining')
flags.DEFINE_float('meta_lr', 0.001, 'Meta learning rate')
flags.DEFINE_integer('validate_every', 500, 'Frequency for metavalidation and saving')
flags.DEFINE_string('savepath', 'saved_models/', 'Path to save or load models')
flags.DEFINE_string('logdir', 'logs/', 'Path to save Tensorboard summaries')

# Testing parameters
flags.DEFINE_integer('num_classes_test', None, 'Number of classes per test task, if different from training')

# Logging parameters
flags.DEFINE_integer('print_every', 100, 'Frequency for printing training loss and accuracy')

# Misc.
flags.DEFINE_string('notes', None, 'Additional notes')
flags.DEFINE_string('filename', None, 'Filename for saving graph')

def main(unused_args):

	hparams = {
		'datasource': FLAGS.datasource,
		'num_classes_train': FLAGS.num_classes,
		'num_classes_val': FLAGS.num_classes,
		'num_classes_test': FLAGS.num_classes_test,
		'num_shot_train': FLAGS.num_shot_train,
		'num_shot_test': FLAGS.num_shot_test,
		'steps': FLAGS.steps,
		'meta_batch_size': FLAGS.meta_batch_size,
		'meta_lr': FLAGS.meta_lr,
		'notes': FLAGS.notes,
	}
	hparams = check_default_config(hparams)
	if FLAGS.train and not FLAGS.load:
		hparams['mode'] = 'train'
		save_string = [
			hparams['datasource'],
			str(hparams['num_classes_train']) + 'way',
			str(hparams['num_shot_train']) + 'shot',
			strftime('%y%m%d_%H%M'),
		]
		save_folder = '_'.join(map(str, save_string)) + '/'
		os.makedirs(FLAGS.savepath + save_folder)
		hparams['savepath'] = FLAGS.savepath + save_folder
		save_config(hparams, FLAGS.savepath + save_folder)
	# elif FLAGS.test:
	# 	hparams = load_config(FLAGS.savepath + 'config.json', test=True, notes=FLAGS.notes)

	if FLAGS.comet:
		experiment = Experiment(api_key=os.environ['COMETML_API_KEY'], project_name='meta')
		experiment.log_multiple_params(hparams)

	if FLAGS.train and FLAGS.datasource in ['omniglot', 'miniimagenet', 'cifar']:

		num_shot_train = FLAGS.num_shot_train or 1
		num_shot_test = FLAGS.num_shot_test or 1

		data_generator = DataGenerator(
			datasource=FLAGS.datasource,
			num_classes=FLAGS.num_classes,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=FLAGS.meta_batch_size,
			test_set=False,
		)

		# Tensorflow queue for metatraining dataset
		# metatrain_image_tensor - (batch_size, num_classes * num_samples_per_class, 28 * 28)
		# metatrain_label_tensor - (batch_size, num_classes * num_samples_per_class, num_classes)
		metatrain_image_tensor, metatrain_label_tensor = data_generator.make_data_tensor(train=True, load=True, savepath='test.pkl')
		train_inputs = tf.slice(metatrain_image_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_inputs = tf.slice(metatrain_image_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metatrain_label_tensor, [0, 0, 0], [-1, FLAGS.num_classes*num_shot_train, -1])
		test_labels = tf.slice(metatrain_label_tensor, [0, FLAGS.num_classes*num_shot_train, 0], [-1, -1, -1])
		metatrain_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		data_generator = DataGenerator(
			datasource=FLAGS.datasource,
			num_classes=hparams['num_classes_val'],
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=16,
			test_set=False,
		)

		# Tensorflow queue for metavalidation dataset
		metaval_image_tensor, metaval_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(metaval_image_tensor, [0, 0, 0], [-1, hparams['num_classes_val']*num_shot_train, -1])
		test_inputs = tf.slice(metaval_image_tensor, [0, hparams['num_classes_val']*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(metaval_label_tensor, [0, 0, 0], [-1, hparams['num_classes_val']*num_shot_train, -1])
		test_labels = tf.slice(metaval_label_tensor, [0, hparams['num_classes_val']*num_shot_train, 0], [-1, -1, -1])
		metaval_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		# Graph for metatraining
		# using scope reuse=tf.AUTO_REUSE, not sure if this is the best way to do it
		if FLAGS.datasource == 'miniimagenet':
			# model_metatrain = CNN_MiniImagenet('model', n_way=FLAGS.num_classes, layers=4, input_tensors=metatrain_input_tensors)
			model_metatrain = CNN_miniimagenet('model', num_classes=FLAGS.num_classes, input_tensors=metatrain_input_tensors)
		elif FLAGS.datasource == 'cifar':
			model_metatrain = CNN_cifar('model', num_classes=FLAGS.num_classes, input_tensors=metatrain_input_tensors)
		else:
			model_metatrain = CNN_omniglot('model', num_classes=FLAGS.num_classes, input_tensors=metatrain_input_tensors)
			# model_metatrain = CNN2('model', n_way=FLAGS.num_classes, layers=4, input_tensors=metatrain_input_tensors)
		
		# Graph for metavalidation
		if FLAGS.datasource == 'miniimagenet':
			# model_metaval = CNN_MiniImagenet('model', n_way=FLAGS.num_classes, layers=4, input_tensors=metaval_input_tensors)
			model_metaval = CNN_miniimagenet('model', num_classes=hparams['num_classes_val'], input_tensors=metaval_input_tensors)
		elif FLAGS.datasource == 'cifar':
			model_metaval = CNN_cifar('model', num_classes=hparams['num_classes_val'], input_tensors=metaval_input_tensors)
		else:
			model_metaval = CNN_omniglot('model', num_classes=FLAGS.num_classes, input_tensors=metaval_input_tensors)
			# model_metaval = CNN2('model', n_way=FLAGS.num_classes, layers=4, input_tensors=metaval_input_tensors)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		if FLAGS.load:
			model_metatrain.load(sess, FLAGS.savepath, verbose=True)
			model_metaval.load(sess, FLAGS.savepath, verbose=True)
		tf.train.start_queue_runners()

		saved_metaval_loss = np.inf
		steps = FLAGS.steps or 40000
		try:
			for step in np.arange(steps):
				# metatrain_loss, metatrain_accuracy, _, _ = sess.run([model_metatrain.loss, model_metatrain.test_accuracy, model_metatrain.optimize, model_metatrain.ae_optimize], {model_metatrain.is_training: True})
				metatrain_loss, metatrain_accuracy, _ = sess.run([model_metatrain.loss, model_metatrain.test_accuracy, model_metatrain.optimize], {model_metatrain.is_training: True})
				if step > 0 and step % FLAGS.print_every == 0:
					# model_metatrain.writer.add_summary(metatrain_summary, step)
					print('Step #{} - Loss : {:.3f} - Acc : {:.3f}'.format(step, metatrain_loss, metatrain_accuracy))
					if FLAGS.comet:
						experiment.log_metric("train_loss", metatrain_loss, step=step)
						experiment.log_metric("train_accuracy", metatrain_accuracy, step=step)
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (steps - 1)):
					if step == (steps - 1):
						print('Training complete!')
					metaval_loss, metaval_accuracy = sess.run([model_metaval.loss, model_metaval.test_accuracy], {model_metaval.is_training: False})
					# model_metaval.writer.add_summary(metaval_summary, step)
					print('Validation Results - Loss : {:.3f} - Acc : {:.3f}'.format(metaval_loss, metaval_accuracy))
					if FLAGS.comet:
						experiment.log_metric("val_loss", metaval_loss, step=step)
						experiment.log_metric("val_accuracy", metaval_accuracy, step=step)
					if metaval_loss < saved_metaval_loss:
						saved_metaval_loss = metaval_loss
						if not FLAGS.load:
							model_metatrain.save(sess, FLAGS.savepath + save_folder, global_step=step, verbose=True)
						else:
							model_metatrain.save(sess, FLAGS.savepath, global_step=step, verbose=True)						
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input('\nSave latest model at Step #{}? (y/n)\n'.format(step))
			if response == 'y':
				model_metatrain.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print('Latest model not saved.')

	if FLAGS.test and FLAGS.datasource in ['omniglot', 'miniimagenet', 'cifar']:

		NUM_TEST_SAMPLES = 600

		num_classes_test = FLAGS.num_classes_test or FLAGS.num_classes

		num_shot_train = FLAGS.num_shot_train or 1
		num_shot_test = FLAGS.num_shot_test or 1

		data_generator = DataGenerator(
			datasource=FLAGS.datasource,
			num_classes=num_classes_test,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=1, # use 1 for testing to calculate stdev and ci95
			test_set=True,
		)

		image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes_test*num_shot_train, -1])
		test_inputs = tf.slice(image_tensor, [0, num_classes_test*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes_test*num_shot_train, -1])
		test_labels = tf.slice(label_tensor, [0, num_classes_test*num_shot_train, 0], [-1, -1, -1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		if FLAGS.datasource == 'miniimagenet':
			# model = CNN_MiniImagenet('model', n_way=FLAGS.num_classes, layers=4, input_tensors=input_tensors)
			model = CNN_miniimagenet('model', num_classes=FLAGS.num_classes, input_tensors=input_tensors)
		elif FLAGS.datasource == 'cifar':
			model = CNN_cifar('model', num_classes=FLAGS.num_classes, input_tensors=input_tensors)
		else:
			model = CNN_omniglot('model', num_classes=FLAGS.num_classes, input_tensors=input_tensors)
			# model = CNN2('model', n_way=FLAGS.num_classes, layers=4, input_tensors=input_tensors)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		model.load(sess, FLAGS.savepath, verbose=True)
		tf.train.start_queue_runners()

		# BEGIN PLOT
		if FLAGS.plot:
			activations, labels = sess.run([model.train_features, model.train_labels], {model.is_training: False})
			activations = activations.reshape([num_shot_train*FLAGS.num_classes, -1])
			from sklearn.manifold import TSNE
			from sklearn.decomposition import PCA
			pca = PCA(50)
			print('Compressing with PCA...')
			activations_50dim = pca.fit_transform(activations)
			tsne = TSNE()
			print('Compressing with tSNE...')
			activations_2dim = tsne.fit_transform(activations_50dim)
			labels = np.argmax(labels, axis=1)
			fig, ax = plt.subplots()
			for i in np.arange(FLAGS.num_classes):
				ax.scatter(activations_2dim[np.where(labels==i)][:, 0], activations_2dim[np.where(labels==i)][:, 1], s=5.)
			plt.show()
			quit()
		# END PLOT

		accuracy_list = []

		for task in np.arange(NUM_TEST_SAMPLES):
			accuracy = sess.run(model.test_accuracy, {model.is_training: False})
			accuracy_list.append(accuracy)
			if task > 0 and task % 100 == 0:
				print('Metatested on {} tasks...'.format(task))

		avg = np.mean(accuracy_list)
		stdev = np.std(accuracy_list)
		ci95 = 1.96 * stdev / np.sqrt(NUM_TEST_SAMPLES)

		print('\nEnd of Test!')
		print('Accuracy                : {:.4f}'.format(avg))
		print('StdDev                  : {:.4f}'.format(stdev))
		print('95% Confidence Interval : {:.4f}'.format(ci95))

		if FLAGS.comet:
			experiment.log_metric("test_accuracy_mean", avg)
			experiment.log_metric("test_accuracy_stdev", stdev)
			experiment.log_metric("test_accuracy_ci95", ci95)

	if FLAGS.train and FLAGS.datasource in ['sinusoid', 'multimodal', 'step']:

		num_shot_train = FLAGS.num_shot_train or 10
		num_shot_test = FLAGS.num_shot_test or 10

		data_generator = DataGenerator(
			datasource=FLAGS.datasource,
			num_classes=None,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=FLAGS.meta_batch_size,
			test_set=None,
		)

		model = FFN('model')
		
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		saved_loss = np.inf
		steps = FLAGS.steps or 50000
		try:
			for step in np.arange(steps):
				if FLAGS.datasource == 'multimodal':
					batch_x, batch_y, amp, phase, slope, intercept, modes = data_generator.generate()
					amp = amp * modes + (modes == False).astype(np.float32)
				elif FLAGS.datasource == 'step':
					batch_x, batch_y, start_step = data_generator.generate()
					amp = np.ones(batch_x.shape[0])
				else:
					batch_x, batch_y, amp, phase = data_generator.generate()
					amp = np.ones(batch_x.shape[0])
				train_inputs = batch_x[:, :num_shot_train, :]
				train_labels = batch_y[:, :num_shot_train, :]
				test_inputs = batch_x[:, num_shot_train:, :]
				test_labels = batch_y[:, num_shot_train:, :]
				feed_dict = {
					model.train_inputs: train_inputs,
					model.train_labels: train_labels,
					model.test_inputs: test_inputs,
					model.test_labels: test_labels,
					model.amp: amp, # use amplitude to scale loss
				}
				
				metatrain_postloss, _ = sess.run([model.loss, model.optimize], feed_dict)
				if step > 0 and step % FLAGS.print_every == 0:
					# model.writer.add_summary(metatrain_summary, step)
					print('Step #{} - PreLoss : {:.3f} - PostLoss : {:.3f}'.format(step, 0., metatrain_postloss))
					if step == (steps - 1):
						print('Training complete!')
					if metatrain_postloss < saved_loss:
						saved_loss = metatrain_postloss
						model.save(sess, FLAGS.savepath + save_folder, global_step=step, verbose=True)
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input('\nSave latest model at Step #{}? (y/n)\n'.format(step))
			if response == 'y':
				model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print('Latest model not saved.')

	if FLAGS.test and FLAGS.datasource in ['sinusoid', 'multimodal', 'step']:

		num_shot_train = FLAGS.num_shot_train or 10

		data_generator = DataGenerator(
			datasource=FLAGS.datasource,
			num_classes=None,
			num_samples_per_class=num_shot_train,
			batch_size=1,
			test_set=None,
		)

		model = FFN('model', num_train_samples=num_shot_train, num_test_samples=50)
		
		sess = tf.InteractiveSession()
		model.load(sess, FLAGS.savepath, verbose=True)

		if FLAGS.datasource == 'multimodal':
			train_inputs, train_labels, amp, phase, slope, intercept, modes = data_generator.generate()
			amp = amp * modes + (modes == False).astype(np.float32)
			x = np.arange(-5., 5., 0.2)
			if modes[0] == 0:
				y = slope * x + intercept
			else:
				y = amp * np.sin(x - phase)
		elif FLAGS.datasource == 'step':
			train_inputs, train_labels, start_step = data_generator.generate()
			x = np.arange(-5., 5., 0.2)
			y = np.ones_like(x) - (x < start_step).astype(np.float32) - (x > (start_step + 2)).astype(np.float32)
		else:
			train_inputs, train_labels, amp, phase = data_generator.generate()
			amp = 5.
			phase = 0.
			x = np.arange(5., 15., 0.2).reshape(1, -1, 1)
			y = amp * np.sin(x - phase).reshape(1, -1, 1)
			train_inputs = np.arange(5., 10., .5).reshape(1, -1, 1)
			# train_inputs = np.arange(-5., 0., .5).reshape(1, -1, 1)
			train_labels = amp * np.sin(train_inputs - phase)

		feed_dict = {
			model.train_inputs: train_inputs,
			model.train_labels: train_labels,
			model.test_inputs: x.reshape(1, -1, 1),
			model.test_labels: y.reshape(1, -1, 1),
		}

		postprediction, postloss = sess.run([model.predictions, model.plain_loss], feed_dict)

		print(postloss)

		fig, ax = plt.subplots()
		ax.plot(x.reshape(-1), y.reshape(-1), color='#2c3e50', linewidth=0.8, label='Truth')
		ax.scatter(train_inputs.reshape(-1), train_labels.reshape(-1), color='#2c3e50', label='Training Set')
		ax.plot(x.reshape(-1), postprediction.reshape(-1), label='Prediction', color='#e74c3c', linestyle='--')
		ax.legend()
		ax.set_title(postloss)
		plt.show()
		if FLAGS.filename is not None:
			fig.savefig('figures/' + FLAGS.filename + '.png', dpi=72, bbox_inches='tight')


if __name__ == '__main__':
	app.run(main)