import os
import pickle
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback

from train_ganomaly import (Decoder, Encoder, FoldLogger, autoencoder_accuracy,
                            autoencoder_false_neg_rate, execution_checks,
                            list_to_text, load_data, load_data_file,
                            make_partition_list,
                            reverse_autoencoder_false_neg_rate, save_data_file,
                            text_to_dataset, verify_partition)

##############
# Parameters #
##############

# Data files
profiles_file = 'data/paper_data/active_meds_list.pkl'
depa_file = 'data/paper_data/depa_list.pkl'

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

# Training parameters
batch_size = 256
max_training_epochs = 1000
single_run_epochs = 21
early_stopping_patience = 5
early_stopping_loss_delta = 0.0001

# Easy names for layers
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
Adam = tf.keras.optimizers.Adam
TextVectorization = tf.keras.layers.experimental.preprocessing.TextVectorization

###########
# Classes #
###########

class AutoEnocder(tf.keras.models.Model):

	def __init__(self, vectorization_layer, autoencoder, name, **kwargs):

		super(AutoEnocder, self).__init__(name=name, **kwargs)

		self.vectorization_layer = vectorization_layer
		self.autoencoder = autoencoder

	def train_step(self, batch):

		x = self.vectorization_layer(tf.expand_dims(batch,1))

		with tf.GradientTape() as tape:

			x_hat = self.autoencoder(x, training=True)

			loss = self.autoencoder.compiled_loss(x, x_hat, regularization_losses=self.autoencoder.losses)

		trainable_vars = self.autoencoder.trainable_variables
		grads = tape.gradient(loss, trainable_vars)
		self.autoencoder.optimizer.apply_gradients(zip(grads, trainable_vars))
		self.autoencoder.compiled_metrics.update_state(x, x_hat)

		return_dict = {m.name: m.result() for m in self.autoencoder.metrics}

		return return_dict
	
	def test_step(self, batch):

		x = self.vectorization_layer(tf.expand_dims(batch,1))
		x_hat = self.autoencoder(x, training=False)

		self.autoencoder.compiled_loss(x, x_hat)
		self.autoencoder.compiled_metrics.update_state(x, x_hat)

		return_dict = {m.name: m.result() for m in self.autoencoder.metrics}

		return return_dict

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

		x = Input(shape=(len(vectorization_layer.get_vocabulary())+1,))
		z = Encoder(autoenc_max_size, autoenc_squeeze_size, dropout, activation_type, name='enc')(x)
		x_hat = Decoder(len(vectorization_layer.get_vocabulary()) + 1, autoenc_max_size, dropout, activation_type, name='dec')(z)

		autoencoder = Model(x, x_hat)
		autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[autoencoder_accuracy, autoencoder_false_neg_rate])
		autoencoder.summary()

		aa = AutoEnocder(vectorization_layer, autoencoder, name='AutoEncoder')
		aa.compile()

		if validate:
			epoch_range = max_training_epochs
		else:
			epoch_range = single_run_epochs

		callbacks = []

		if validate:
			callbacks.append(tf.keras.callbacks.EarlyStopping(
				monitor="val_loss",
				min_delta=early_stopping_loss_delta,
				patience=early_stopping_patience,
				verbose=2, restore_best_weights=True
				))
		callbacks.append(FoldLogger(fold))
		callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir,'tensorboard', str(fold))))
		callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(save_dir,'training.csv'), append=True))
		callbacks.append(TqdmCallback(verbose=1, data_size=len(profiles_train), batch_size=batch_size))

		aa.fit(train_ds, epochs=epoch_range, validation_data = val_ds, verbose=0, callbacks=callbacks)

		if validate == False:
			autoencoder.save(os.path.join(save_dir, 'trained_model'))
			save_data_file(os.path.join(save_dir,'trained_model', 'vocabulary.pkl'), vectorization_layer.get_vocabulary())
