import os
import pickle
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

##############
# Parameters #
##############

# Data files
profiles_file = 'data/paper_data/train/active_meds_list.pkl'
depa_file = 'data/paper_data/train/depa_list.pkl'

# Save dir
save_dir = 'model'

# Years to use
train_years_begin = [2005,2006,2007] # inclusively
train_years_end = [2014,2015,2016]# inclusively
val_years_begin = [2015,2016,2017] # inclusively
val_years_end = [2015,2016,2017] # inclusively

# Model parameters
autoenc_max_size = 256
autoenc_squeeze_size = 64
dropout = 0.1
activation_type = 'SELU'
feat_ext_max_size = 128
feat_ext_min_size = 64
loss_weights = [100,2,1] # in order: contextual loss, adversarial loss, encoder loss
disc_lr = 1e-6
l1l2ratio = 0.8

# Training parameters
batch_size = 256
max_training_epochs = 1000
single_run_epochs = 21
early_stopping_patience = 5
early_stopping_loss_delta = 0.001

# Easy names for layers
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
BatchNormalization = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU
Adam = tf.keras.optimizers.Adam
TextVectorization = tf.keras.layers.experimental.preprocessing.TextVectorization

###########
# Classes #
###########

# Custom encoder layer
class Encoder(tf.keras.layers.Layer):
	
	def __init__(self, autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name, **kwargs):
		super(Encoder, self).__init__(name=name, **kwargs)
		self.autoenc_max_size = autoenc_max_size
		self.autoenc_squeeze_size = autoenc_squeeze_size
		self.dropout = dropout
		if activation_type == 'ReLU':
			self.activation = 'relu'
			self.initializer = 'glorot_uniform'
		elif activation_type == 'SELU':
			self.activation = 'selu'
			self.initializer = 'lecun_normal'

		self.encoderl1 = Dense(self.autoenc_max_size, activation=self.activation, kernel_initializer=self.initializer)
		self.encoderl2 = Dropout(self.dropout)
		self.encoderl3 = Dense(self.autoenc_squeeze_size)

	def call(self, inputs):

		# Encoder
		return self.encoderl3(self.encoderl2(self.encoderl1(inputs)))

	# Make it serializeable
	def get_config(self):
		return {
			'autoenc_max_size':self.autoenc_max_size,
			'autoenc_squeeze_size':self.autoenc_squeeze_size,
			'dropout':self.dropout,
			'activation':self.activation,
			'initializer':self.initializer,
			'name':self.name
		}

# Custom decoder layer
class Decoder(tf.keras.layers.Layer):

	def __init__(self, output_dim, autoenc_max_size, dropout, activation_type, name, **kwargs):
		super(Decoder, self).__init__(name=name, **kwargs)
		self.output_dim = output_dim
		self.autoenc_max_size = autoenc_max_size
		self.dropout = dropout
		if activation_type == 'ReLU':
			self.activation = 'relu'
			self.initializer = 'glorot_uniform'
		elif activation_type == 'SELU':
			self.activation = 'selu'
			self.initializer = 'lecun_normal'
		self.decoderl1 = Dense(self.autoenc_max_size, activation=self.activation, kernel_initializer=self.initializer)
		self.decoderl2 = Dropout(self.dropout)
		self.decoderl3 = Dense(self.output_dim, activation='sigmoid')

	def call(self, inputs):

		# Decoder
		return self.decoderl3(self.decoderl2(self.decoderl1(inputs)))

	# Make it serializeable
	def get_config(self):
		return {
			'output_dim':self.output_dim,
			'autoenc_max_size':self.autoenc_max_size,
			'dropout':self.dropout,
			'activation':self.activation,
			'initializer':self.initializer,
			'name':self.name
		}

class FeatureExtractor(tf.keras.layers.Layer):

	def __init__(self, feat_ext_max_size, feat_ext_min_size, dropout, name, **kwargs):
		super(FeatureExtractor, self).__init__(name=name, **kwargs)
		self.feat_ext_max_size = feat_ext_max_size
		self.feat_ext_min_size = feat_ext_min_size
		self.dropout = dropout
		self.featextl1 = Dense(self.feat_ext_max_size)
		self.featextl2 = ReLU()
		self.featextl3 = BatchNormalization()
		self.featextl4 = Dropout(self.dropout)
		self.featextl5 = Dense(self.feat_ext_min_size)

	def call(self, inputs):

		return self.featextl5(self.featextl4(self.featextl3(self.featextl2(self.featextl1(inputs)))))

	def get_config(self):
		return {
			'feat_ext_max_size':self.feat_ext_max_size,
			'feat_ext_min_size':self.feat_ext_min_size,
			'dropout':self.dropout,
			'name':self.name
		}

class GanContinueChecker:

	def __init__(self):
		self.checks = {
			'early_stopping_check':{
				'patience':early_stopping_patience,
				'min_delta':early_stopping_loss_delta,
				'reporting_string':'Loss decreased less than {} over {} epochs, stopping training.\n\n',
				'trigger_result': 'early_stop',
				}, 
			}
		self.absolute_min_loss = 999999
		self.absolute_min_loss_epoch = 0

	def gan_continue_check(self, val_monitor_losses, epoch):
		return_object = []
		cur_epoch_loss = val_monitor_losses[-1]
		if cur_epoch_loss < self.absolute_min_loss:
			self.absolute_min_loss = cur_epoch_loss
			self.absolute_min_loss_epoch = epoch
		for _, check_dict in self.checks.items():
			if len(val_monitor_losses) < check_dict['patience'] + 1:
				continue
			if cur_epoch_loss > (self.absolute_min_loss - check_dict['min_delta']) and epoch > (self.absolute_min_loss_epoch + check_dict['patience']):
				print(check_dict['reporting_string'].format(check_dict['min_delta'], check_dict['patience']))
				return_object.append(check_dict['trigger_result'])
		print('Current epoch monitored loss: {:.5f}'.format(cur_epoch_loss))
		print('Absolute minimum loss: {:.5f} at epoch {}\n\n'.format(self.absolute_min_loss, self.absolute_min_loss_epoch + 1))
		return return_object

#############
# Functions #
#############

def execution_checks(save_dir, train_or_test_years_begin, train_or_test_years_end, val_years_begin = None, val_years_end = None):

	if val_years_begin == None:
		val_years_begin = []
	if val_years_end == None:
		val_years_end = []

	if os.path.exists(save_dir):
		print('ERROR: model folder already exists. Move it or delete it before continuing.')
		quit()

	try:
		assert len(train_or_test_years_begin) == len(train_or_test_years_end)
		if len(val_years_begin) > 0:
			assert len(train_or_test_years_begin) == len(train_or_test_years_end) == len(val_years_begin) == len(val_years_end)
	except:
		print('ERROR: number of years in years to use as boundaries for data partitions is not equal.')
		quit()

	if len(val_years_begin) == 0:
		validate = False
		try:
			assert len(train_or_test_years_begin) == 1
		except:
			print('ERROR: More than 1 training data partition defined but no validation set defined. Cannot perform cross-validation without validation partition.')
			quit()
	else:
		validate = True

	n_cross_val_folds = len(train_or_test_years_begin)

	train_or_test_years_begin = iter(train_or_test_years_begin)
	train_or_test_years_end = iter(train_or_test_years_end)
	val_years_begin = iter(val_years_begin)
	val_years_end = iter(val_years_end)

	return validate, n_cross_val_folds, train_or_test_years_begin, train_or_test_years_end, val_years_begin, val_years_end

def load_data(profiles_file, depa_file):
	
	print('Loading data...')

	profiles = load_data_file(profiles_file)
	depa = load_data_file(depa_file)

	return profiles, depa

def load_data_file(filepath):
	with open(filepath, mode='rb') as opened_file:
		loaded_file = pickle.load(opened_file)
	return loaded_file

def save_data_file(filepath, data):
	with open(filepath, mode='wb') as opened_file:
		pickle.dump(data,opened_file)

def make_partition_list(source_list, year_begin, year_end):
	return list(chain.from_iterable([source_list[year] for year in range(year_begin, year_end + 1)]))

def empty_list_idxes(list_to_examine):
	return [idx for idx, element in enumerate(list_to_examine) if len(element) == 0]

def remove_by_idx(list_to_clean, idxes):
	for idx in sorted(idxes, reverse=True):
		list_to_clean.pop(idx)
	return list_to_clean

def not_in_dict_keys_idxes(list_to_examine, dictionary, list_in_element=True):
	if list_in_element:
		return [idx for idx, element in enumerate(list_to_examine) if element[0] not in dictionary.keys()]
	else:
		return [idx for idx, element in enumerate(list_to_examine) if element not in dictionary.keys()]

def verify_partition(profiles_list, depa_list, partition_name):
	print('Verifying {} data partition...'.format(partition_name))

	print('Number of samples in profiles: {}'.format(len(profiles_list)))
	print('Number of samples in depa: {}'.format(len(depa_list)))

	try:
		assert len(profiles_list) == len(depa_list)
	except:
		print('ERROR: Number of samples in  partition is not equal between profiles and departements')
		quit()
	
	return

def list_to_text(list_to_join):
	return [' '.join([str(el) for el in l]) for l in list_to_join]

def text_to_dataset(text, shuffle=False):
	ds = tf.data.Dataset.from_tensor_slices(text)
	if shuffle:
		ds = ds.shuffle(buffer_size=len(text))
	return ds

def encoder_loss(y_true, y_pred):
	tf_l1l2ratio = tf.dtypes.cast(tf.constant(l1l2ratio), tf.float32)
	l1loss = tf.keras.losses.MAE(y_true, y_pred)
	l2loss = tf.keras.losses.MSE(y_true, y_pred)
	l1l2loss =  tf.math.add(tf.math.multiply(tf_l1l2ratio, l1loss), tf.math.multiply(tf.math.subtract(tf.dtypes.cast(tf.constant(1), tf.float32), tf_l1l2ratio), l2loss))
	return l1l2loss

def autoencoder_accuracy(y_true, y_pred):
	dichot_true = tf.dtypes.cast(tf.math.greater_equal(y_true, tf.constant(0.5)), tf.float32)
	dichot_ypred = tf.dtypes.cast(tf.math.greater_equal(y_pred, tf.constant(0.5)), tf.float32)
	maximums = tf.math.count_nonzero(tf.math.maximum(dichot_true, dichot_ypred), 1, dtype=tf.float32)
	correct = tf.math.count_nonzero(tf.math.multiply(dichot_true, dichot_ypred), 1, dtype=tf.float32)
	return tf.math.xdivy(correct, maximums)

def autoencoder_false_neg_rate(y_true, y_pred):
	dichot_y_true = tf.dtypes.cast(tf.math.greater_equal(y_true, tf.constant(0.5)), tf.float32)
	dichot_ypred = tf.dtypes.cast(tf.math.greater_equal(y_pred, tf.constant(0.5)), tf.float32)
	true = tf.math.count_nonzero(dichot_y_true, 1, dtype=tf.float32)
	correct = tf.math.count_nonzero(tf.math.multiply(dichot_y_true, dichot_ypred), 1, dtype=tf.float32)
	false_negs = tf.math.subtract(true, correct)
	return tf.math.xdivy(false_negs, true)

def reverse_autoencoder_false_neg_rate(y_true, y_pred):
	dichot_y_true = tf.dtypes.cast(tf.math.greater_equal(y_true, tf.constant(0.5)), tf.float32)
	dichot_ypred = tf.dtypes.cast(tf.math.greater_equal(y_pred, tf.constant(0.5)), tf.float32)
	# Difference is in following line
	true = tf.math.count_nonzero(dichot_ypred, 1, dtype=tf.float32)
	correct = tf.math.count_nonzero(tf.math.multiply(dichot_y_true, dichot_ypred), 1, dtype=tf.float32)
	false_negs = tf.math.subtract(true, correct)
	return tf.math.xdivy(false_negs, true)

###########
# Execute #
###########

if __name__ == '__main__':

	# Check that the provided years are okay and define execution mode

	validate, n_cross_val_folds, train_years_begin, train_years_end, val_years_begin, val_years_end = execution_checks(save_dir, train_years_begin, train_years_end, val_years_begin, val_years_end)

	# Load data

	profiles, depa = load_data(profiles_file, depa_file)

	# Train

	for fold in range(n_cross_val_folds):

		if n_cross_val_folds > 1:
			print('CROSS-VALIDATION FOLD {}\n\n'.format(fold + 1))

		# Divide into train and val

		train_year_begin = next(train_years_begin)
		train_year_end = next(train_years_end)
		if validate:
			val_year_begin = next(val_years_begin)
			val_year_end = next(val_years_end)

		profiles_train = make_partition_list(profiles, train_year_begin, train_year_end)
		depa_train = make_partition_list(depa, train_year_begin, train_year_end)
		verify_partition(profiles_train, depa_train, 'train')
		if validate:
			profiles_val = make_partition_list(profiles, val_year_begin, val_year_end)
			depa_val = make_partition_list(depa, val_year_begin, val_year_end)
			verify_partition(profiles_val, depa_val, 'val')

		# Convert the lists to Tensorflow Datasets, shuffle and batch

		train_ds = text_to_dataset(list_to_text(profiles_train), shuffle=True).batch(batch_size).prefetch(25)
		if validate:
			val_ds = text_to_dataset(list_to_text(profiles_val)).batch(batch_size).prefetch(25)
		else:
			val_ds = None

		# Instantiate the model and prepare the variables
		
		vectorization_layer = TextVectorization(output_mode='binary')
		vectorization_layer.adapt(train_ds)

		x = Input(shape=(len(vectorization_layer.get_vocabulary()),))
		z = Encoder(autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name='enc1')(x)
		x_hat = Decoder(len(vectorization_layer.get_vocabulary()), autoenc_max_size, dropout, activation_type, name='dec')(z)
		z_hat = Encoder(autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name='enc2')(x_hat)
		
		feature_extracted = FeatureExtractor(feat_ext_max_size, feat_ext_min_size, dropout, name='feat_ext')(x)
		disc = ReLU()(feature_extracted)
		disc = Dense(1, activation='sigmoid')(disc)
		
		discriminator = Model(x, disc, name='discriminator')
		discriminator.compile(optimizer=Adam(learning_rate=disc_lr), loss='binary_crossentropy', metrics='accuracy')
		discriminator.summary()

		discriminator.trainable = False
		feature_extracted_from_x = discriminator.get_layer('feat_ext')(x)

		adversarial_autoencoder = Model(x, [x_hat, feature_extracted_from_x, z_hat], name='adversarial_autoencoder')
		adversarial_autoencoder.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse', encoder_loss], metrics=[[autoencoder_accuracy, autoencoder_false_neg_rate], None, None], loss_weights=loss_weights)
		
		adversarial_autoencoder.summary()

		# Train

		if validate:
			epoch_range = max_training_epochs
		else:
			epoch_range = single_run_epochs

		# Custom training loop
		c = GanContinueChecker()
		val_monitor_losses = []
		for epoch in range(epoch_range):

			length_train = len(list(train_ds.as_numpy_iterator()))
			train_dsi = train_ds.as_numpy_iterator()
			length_val = len(list(val_ds.as_numpy_iterator()))
			val_dsi = val_ds.as_numpy_iterator()

			d_losses = []
			g_losses = []

			print('EPOCH {}\n\n'.format(epoch + 1))
			print('TRAINING...')

			for batch_data_X in tqdm(train_dsi, total=length_train):
			
				# Train discriminator

				ones_label = np.ones((len(batch_data_X),1))
				zeros_label = np.zeros((len(batch_data_X),1))

				multi_hot = vectorization_layer(tf.expand_dims(batch_data_X, 1))
				latent_from_data = adversarial_autoencoder.get_layer('enc1')(multi_hot, training=False)
				reconstructed_from_data = adversarial_autoencoder.get_layer('dec')(latent_from_data, training=False)

				d_loss_generated = discriminator.train_on_batch(reconstructed_from_data, ones_label)
				d_loss_from_data = discriminator.train_on_batch(multi_hot, zeros_label)
				d_loss = 0.5 * np.add(d_loss_generated, d_loss_from_data)

				# Train generator

				discriminator.trainable = False

				feature_extracted = discriminator.get_layer('feat_ext')(reconstructed_from_data, training=False)
				g_loss = adversarial_autoencoder.train_on_batch(multi_hot, [multi_hot, feature_extracted, latent_from_data])

				discriminator.trainable = True

				d_losses.append(d_loss)
				g_losses.append(g_loss)

			# Compute the metrics for the epoch
			d_losses = np.array(d_losses)
			d_losses = np.mean(d_losses, axis=0)

			g_losses = np.array(g_losses)
			g_losses = np.mean(g_losses, axis=0)

			all_names = discriminator.metrics_names + adversarial_autoencoder.metrics_names
			all_losses = np.hstack((d_losses, g_losses)).tolist()

			print('EPOCH {} TRAINING RESULTS:'.format(epoch+1))
			print('\n'.join(['{} : {:.5f}'.format(name,metric) for name,metric in zip(all_names, all_losses)]))
			print('\n')

			g_val_losses = []
			if validate:

				print('VALIDATION...')
				for batch_data_X  in tqdm(val_dsi, total=length_val):

					multi_hot = vectorization_layer(tf.expand_dims(batch_data_X, 1))
					ones_label = np.ones((len(batch_data_X),1))

					latent_repr = adversarial_autoencoder.get_layer('enc1')(multi_hot, training=False)
					reconstructed_from_data = adversarial_autoencoder.get_layer('dec')(latent_repr, training=False)
					feature_extracted = discriminator.get_layer('feat_ext')(reconstructed_from_data, training=False)

					g_loss = adversarial_autoencoder.test_on_batch(multi_hot, [multi_hot, feature_extracted, latent_repr])

					g_val_losses.append(g_loss)

				g_val_losses = np.array(g_val_losses)
				g_val_losses = np.mean(g_val_losses, axis=0)
				val_monitor_losses.append(g_val_losses[0])
				val_names = ['val_' + name for name in adversarial_autoencoder.metrics_names]

				print('EPOCH {} VALIDATION RESULTS:'.format(epoch+1))
				print('\n'.join(['{} : {:.5f}'.format(name,metric) for name,metric in zip(val_names, g_val_losses)]))
				print('\n')

				all_names = all_names + val_names
				all_losses = np.hstack((all_losses, g_val_losses)).tolist()

			continue_check = c.gan_continue_check(val_monitor_losses,epoch)
			if 'early_stop' in continue_check:
				break

			epoch_results_df = pd.DataFrame.from_dict({epoch: dict(zip(all_names, all_losses))}, orient='index')
			epoch_results_df['epoch']=epoch
			epoch_results_df['fold']=fold
			# save the dataframe to csv file, create new file at first epoch, else append
			if epoch == 0 and fold == 0:
				epoch_results_df.to_csv(os.path.join('training_history.csv'))
			else:
				epoch_results_df.to_csv(os.path.join(
					'training_history.csv'), mode='a', header=False)

	if validate == False:
		adversarial_autoencoder.save(os.path.join('trained_model'))
		save_data_file(os.path.join('trained_model', 'vocabulary.pkl'), vectorization_layer.get_vocabulary())





