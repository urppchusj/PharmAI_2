import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.filters import threshold_otsu
from tqdm import tqdm

from train_ganomaly import (Decoder, Encoder, FeatureExtractor,
                            autoencoder_accuracy, autoencoder_false_neg_rate,
                            empty_list_idxes, encoder_loss, list_to_text,
                            load_data, load_data_file, make_partition_list,
                            remove_by_idx, reverse_autoencoder_false_neg_rate,
                            save_data_file, text_to_dataset, verify_partition)

##############
# Parameters #
##############

# Files and dirs
model_dir = 'model'
profiles_file = 'data/paper_data/test/active_meds_list.pkl'
depa_file = 'data/paper_data/test/depa_list.pkl'

# Years to use
test_years_begin = 2018 # inclusively
test_years_end = 2018 # inclusively

# Model parameters
batch_size = 256
l1l2ratio = 0.8

# Easy names for layers
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
TextVectorization = tf.keras.layers.experimental.preprocessing.TextVectorization

#############
# Functions #
#############

###########
# Execute #
###########

if __name__ == '__main__':

	profiles, depa = load_data(profiles_file, depa_file)

	profiles_test = make_partition_list(profiles, test_years_begin, test_years_end)
	depa_test = make_partition_list(depa, test_years_begin, test_years_end)

	# Clean test set of empty departements
	no_depa_idxes = empty_list_idxes(depa_test)
	profiles_test = remove_by_idx(profiles_test, no_depa_idxes)
	depa_test = remove_by_idx(depa_test, no_depa_idxes)
	verify_partition(profiles_test, depa_test, 'test')
	cat_depa_test = [d[0] for d in depa_test]
	save_data_file(os.path.join(model_dir, 'cat_depa_list.pkl'), cat_depa_test)
	unique_categorized_departements = list(set(cat_depa_test))

	test_ds = text_to_dataset(list_to_text(profiles_test)).batch(batch_size).prefetch(25)
	
	adv_autoencoder = tf.keras.models.load_model(os.path.join(model_dir, 'trained_model.h5'), custom_objects={'Encoder':Encoder, 'Decoder':Decoder, 'FeatureExtractor':FeatureExtractor, 'encoder_loss':encoder_loss, 'autoencoder_accuracy':autoencoder_accuracy, 'autoencoder_false_neg_rate':autoencoder_false_neg_rate})
	vocabulary = load_data_file(os.path.join(model_dir, 'vocabulary.pkl'))
	vectorization_layer = TextVectorization(output_mode='binary')
	vectorization_layer.set_vocabulary(vocabulary[1:])

	x, x_hat, z, z_hat, fe = [], [], [], [], []
	for idx, batch in enumerate(tqdm(test_ds)):
		batch_x = vectorization_layer(batch)
		batch_z = adv_autoencoder.get_layer('enc1')(batch_x, training=False)
		batch_x_hat, batch_fe, batch_z_hat = adv_autoencoder.predict_on_batch(batch_x)
		x.append(batch_x)
		x_hat.append(batch_x_hat)
		z.append(batch_z)
		z_hat.append(batch_z_hat)
		fe.append(batch_fe) 

	x = np.vstack(x)
	x_hat = np.vstack(x_hat)	
	z = np.vstack(z)
	z_hat = np.vstack(z_hat)
	fe = np.vstack(fe)

	results = adv_autoencoder.evaluate(x, [x, fe, z], verbose=1)
	joblib.dump(results, os.path.join(model_dir, 'results.joblib'))

	x_hat_dichot = (x_hat >= 0.5) * 1
	joblib.dump(x, os.path.join(model_dir, 'eval_true.joblib'))
	joblib.dump(x_hat, os.path.join(model_dir, 'eval_predictions.joblib'))
	
	with open(os.path.join(model_dir,'evaluation_results.txt'), mode='w', encoding='cp1252') as file:
		file.write('Evaluation on test set results\n')
		file.write('Predictions for {} classes\n'.format(len(vocabulary)))
		file.write('{} classes reprensented in targets\n'.format(sum(np.amax(x_hat_dichot, axis=0))))
		for metric, result in zip(adv_autoencoder.metrics_names, results):
			file.write('Metric: {}   Score: {:.5f}\n'.format(metric,result))
	
	encoder_losses = encoder_loss(z, z_hat)
	encoder_losses = encoder_losses.numpy()
	joblib.dump(encoder_losses, os.path.join(model_dir, 'encoder_losses.joblib'))

	# Calculate the otsu thresholds per departement for use on subsequent data
	otsu_dict = {}

	global_encloss_forotsu = encoder_losses[encoder_losses < np.percentile(encoder_losses,90)]
	global_otsu = threshold_otsu(global_encloss_forotsu)
	otsu_dict['global'] = global_otsu
	for depa in unique_categorized_departements:
	
		depa_idxes = [idx for idx, element in enumerate(depa_test) if element[0] == depa]
		depa_encloss = np.take(encoder_losses, depa_idxes)
		
		depa_encloss_forotsu = depa_encloss[depa_encloss < np.percentile(depa_encloss,90)]
		otsu = threshold_otsu(depa_encloss_forotsu)

		otsu_dict[depa] = otsu

	save_data_file(os.path.join(model_dir, 'otsu_thresholds.pkl'), otsu_dict)
