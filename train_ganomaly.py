import os
import pickle
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback

##############
# Parameters #
##############

# Data files
profiles_file = 'data/paper_data/active_meds_list.pkl'
depa_file = 'data/paper_data/depa_list.pkl'
depa_dict_file = 'data/paper_data/depas.csv'

# Save dir
save_dir = 'model'

# Years to use
train_years_begin = [2014,2015,2016,2013,2014,2015,2012,2013,2014,2011,2012,2013,2010,2011,2012,2009,2010,2011,2008,2009,2010,2007,2008,2009,2006,2007,2008,2005,2006,2007] # inclusively
train_years_end = [2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016,2014,2015,2016]# inclusively
val_years_begin = [2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017] # inclusively
val_years_end = [2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017,2015,2016,2017] # inclusively

# Model parameters
autoenc_max_size = 256
autoenc_squeeze_size = 64
dropout = 0.1
activation_type = 'SELU'
feat_ext_max_size = 128
feat_ext_min_size = 64
loss_weights = [100,1,1] # in order: contextual loss, adversarial loss, encoder loss
disc_lr = 1e-6
l1l2ratio = 0.8

# Training parameters
batch_size = 256
max_training_epochs = 1000
single_run_epochs = 215
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

# Define the ganomaly model
class Ganomaly(tf.keras.Model):

	def __init__(self, vectorization_layer, adversarial_autoencoder, discriminator, name, **kwargs):
		
		super(Ganomaly, self).__init__(name=name, **kwargs)
		self.vectorization_layer = vectorization_layer
		self.adversarial_autoencoder = adversarial_autoencoder
		self.discriminator = discriminator

	def train_step(self, batch):

		# Train discriminator

		x = self.vectorization_layer(tf.expand_dims(batch, 1))
		z = self.adversarial_autoencoder.get_layer('enc1')(x, training=False)
		x_hat = self.adversarial_autoencoder.get_layer('dec')(z, training=False)

		ones_label = tf.ones((tf.shape(x)[0],1))
		zeros_label = tf.zeros((tf.shape(x)[0],1))
		y = tf.concat([ones_label, zeros_label], axis=0)

		with tf.GradientTape() as tape:

			y_pred_from_x_hat = self.discriminator(x_hat, training=True)
			y_pred_from_x = self.discriminator(x, training=True)

			y_pred = tf.concat([y_pred_from_x_hat, y_pred_from_x], axis=0)

			d_loss = self.discriminator.compiled_loss(y, y_pred, regularization_losses=self.discriminator.losses)

		trainable_vars = self.discriminator.trainable_variables
		grads = tape.gradient(d_loss, trainable_vars)
		self.discriminator.optimizer.apply_gradients(zip(grads, trainable_vars))
		self.discriminator.compiled_metrics.update_state(y, y_pred)

		# Train autoencoder

		self.discriminator.trainable = False
		
		feature_extracted_from_x_hat = self.discriminator.get_layer('feat_ext')(x_hat, training=False)
		
		with tf.GradientTape() as tape:

			x_hat, feature_extracted_from_x, z_hat = self.adversarial_autoencoder(x, training=True)
			z = self.adversarial_autoencoder.get_layer('enc1')(x, training=True)

			a_loss = self.adversarial_autoencoder.compiled_loss([x, feature_extracted_from_x_hat, z], [x_hat, feature_extracted_from_x, z_hat], regularization_losses=self.adversarial_autoencoder.losses)

		trainable_vars = self.adversarial_autoencoder.trainable_variables
		grads = tape.gradient(a_loss, trainable_vars)
		self.adversarial_autoencoder.optimizer.apply_gradients(zip(grads, trainable_vars))
		self.adversarial_autoencoder.compiled_metrics.update_state([x, feature_extracted_from_x_hat, z], [x_hat, feature_extracted_from_x, z_hat])

		self.discriminator.trainable = True
		
		return_dict = {**{'A_' + m.name: m.result() for m in self.adversarial_autoencoder.metrics}, **{'D_' + m.name: m.result() for m in self.discriminator.metrics}}

		return return_dict

	def test_step(self, batch):

		x = self.vectorization_layer(tf.expand_dims(batch, 1))
		x_hat, feature_extracted_from_x, z_hat = self.adversarial_autoencoder(x, training=False)
		z = self.adversarial_autoencoder.get_layer('enc1')(x, training=False)
		feature_extracted_from_x_hat = self.discriminator.get_layer('feat_ext')(x_hat, training=False)

		self.adversarial_autoencoder.compiled_loss([x, feature_extracted_from_x_hat, z], [x_hat, feature_extracted_from_x, z_hat], regularization_losses=self.adversarial_autoencoder.losses)
		self.adversarial_autoencoder.compiled_metrics.update_state([x, feature_extracted_from_x_hat, z], [x_hat, feature_extracted_from_x, z_hat])

		return_dict = {'A_' + m.name: m.result() for m in self.adversarial_autoencoder.metrics}

		return return_dict

# Custom callback to log the fold
class FoldLogger(tf.keras.callbacks.Callback):

	def __init__(self, fold, *args, **kwargs):
		super(FoldLogger, self).__init__(*args, **kwargs)
		self.fold=fold

	def on_epoch_end(self, epoch, logs):
		logs['fold'] = self.fold

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

def load_data(profiles_file, depa_file, depa_dict_file):
	
	print('Loading data...')

	profiles = load_data_file(profiles_file)
	depa = load_data_file(depa_file)

	depa_df = pd.read_csv(depa_dict_file, sep=';')
	depa_dict = depa_df.dropna(subset=['cat_depa']).set_index('orig_depa')['cat_depa'].to_dict()

	return profiles, depa, depa_dict

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

def filter_partition(profiles_list, depa_list, depa_dict, partition_name):
	print('Building {} data partition...'.format(partition_name))

	print('Number of samples in profiles: {}'.format(len(profiles_list)))
	print('Number of samples in depa: {}'.format(len(depa_list)))

	try:
		assert len(profiles_list) == len(depa_list)
	except:
		print('ERROR: Number of samples in  partition is not equal between profiles and departements')
		quit()

	# Remove samples with no depa (interferes with categorization by department later on)

	print('Dropping samples with no department...')

	samples_idx_todrop = empty_list_idxes(depa_list)
	profiles_list = remove_by_idx(profiles_list, samples_idx_todrop)
	depa_list = remove_by_idx(depa_list, samples_idx_todrop)
	print('Dropped {} samples in set'.format(len(samples_idx_todrop)))
	print('Number of samples in profiles: {}'.format(len(profiles_list)))
	print('Number of samples in depa: {}'.format(len(depa_list)))

	# Remove departments not able to be categorized

	print('Dropping samples with uncategorized departments...')

	samples_idx_todrop = not_in_dict_keys_idxes(depa_list, depa_dict)
	profiles_list = remove_by_idx(profiles_list, samples_idx_todrop)
	depa_list = remove_by_idx(depa_list, samples_idx_todrop)
	print('Dropped {} samples in set'.format(len(samples_idx_todrop)))
	print('Number of samples in profiles: {}'.format(len(profiles_list)))
	print('Number of samples in depa: {}'.format(len(depa_list)))

	return profiles_list, depa_list

def list_to_text(list_to_join):
	return [' '.join(l) for l in list_to_join]

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

	profiles, depa, depa_dict = load_data(profiles_file, depa_file, depa_dict_file)

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
		profiles_train, depa_train = filter_partition(profiles_train, depa_train, depa_dict, 'train')
		if validate:
			profiles_val = make_partition_list(profiles, val_year_begin, val_year_end)
			depa_val = make_partition_list(depa, val_year_begin, val_year_end)
			profiles_val, depa_val = filter_partition(profiles_val, depa_val, depa_dict, 'val')

		# Convert the lists to Tensorflow Datasets, shuffle and batch

		train_ds = text_to_dataset(list_to_text(profiles_train), shuffle=True).batch(batch_size).prefetch(25)
		if validate:
			val_ds = text_to_dataset(list_to_text(profiles_val)).batch(batch_size).prefetch(25)
		else:
			val_ds = None

		# Instantiate the model and prepare the variables
		
		vectorization_layer = TextVectorization(output_mode='binary')
		vectorization_layer.adapt(train_ds)

		x = Input(shape=(len(vectorization_layer.get_vocabulary()) + 1,))
		z = Encoder(autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name='enc1')(x)
		x_hat = Decoder(len(vectorization_layer.get_vocabulary()) + 1, autoenc_max_size, dropout, activation_type, name='dec')(z)
		z_hat = Encoder(autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name='enc2')(x_hat)
		
		feature_extracted = FeatureExtractor(feat_ext_max_size, feat_ext_min_size, dropout, name='feat_ext')(x)
		disc = ReLU()(feature_extracted)
		disc = BatchNormalization()(disc)
		disc = Dropout(dropout)(disc)
		disc = Dense(1, activation='sigmoid')(disc)
		
		discriminator = Model(x, disc, name='discriminator')
		discriminator.compile(optimizer=Adam(learning_rate=disc_lr), loss='binary_crossentropy', metrics='accuracy')

		feature_extracted_from_x = discriminator.get_layer('feat_ext')(x)

		adversarial_autoencoder = Model(x, [x_hat, feature_extracted_from_x, z_hat], name='adversarial_autoencoder')
		adversarial_autoencoder.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse', encoder_loss], metrics=[[autoencoder_accuracy, autoencoder_false_neg_rate], None, None], loss_weights=loss_weights)
		
		discriminator.summary()
		adversarial_autoencoder.summary()
		g = Ganomaly(vectorization_layer, adversarial_autoencoder, discriminator, name='ganomaly')

		g.compile()

		# Train

		if validate:
			epoch_range = max_training_epochs
		else:
			epoch_range = single_run_epochs

		callbacks = []

		if validate:
			callbacks.append(tf.keras.callbacks.EarlyStopping(
				monitor="val_A_loss",
				min_delta=early_stopping_loss_delta,
				patience=early_stopping_patience,
				verbose=2, restore_best_weights=True
				))
		callbacks.append(FoldLogger(fold))
		callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir,'tensorboard', str(fold))))
		callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(save_dir,'training.csv'), append=True))
		callbacks.append(TqdmCallback(verbose=1, data_size=len(profiles_train), batch_size=batch_size))
		
		g.fit(train_ds, epochs=epoch_range, validation_data=val_ds, verbose=0, callbacks=callbacks)

	if validate == False:
		g.adversarial_autoencoder.save(os.path.join(save_dir, 'trained_model'))
		save_data_file(os.path.join(save_dir,'trained_model', 'vocabulary.pkl'), vectorization_layer.get_vocabulary())





