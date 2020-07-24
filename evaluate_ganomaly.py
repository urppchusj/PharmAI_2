import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from skimage.filters import threshold_otsu
from train_ganomaly import (Ganomaly, autoencoder_accuracy,
                            autoencoder_false_neg_rate, encoder_loss,
                            filter_partition, list_to_text, load_data,
                            load_data_file, make_partition_list,
                            reverse_autoencoder_false_neg_rate, save_data_file,
                            text_to_dataset)

##############
# Parameters #
##############

# Files and dirs
model_dir = 'ganomaly/paper-final-model/trained_model'
profiles_file = 'data/paper_data/active_meds_list_test.pkl'
depa_file = 'data/paper_data/depa_list_test.pkl'
depa_dict_file = 'data/paper_data/depas.csv'

# Years to use
test_years_begin = 2018 # inclusively
test_years_end = 2018 # inclusively

# Model parameters
batch_size = 256
l1l2ratio = 0.8

# Easy names for layers
Input = tf.keras.layers.Input
Lambda = tf.keras.layers.Lambda
Model = tf.keras.models.Model
TextVectorization = tf.keras.layers.experimental.preprocessing.TextVectorization

#############
# Functions #
#############

def build_inference_ganomaly(vocabulary, adv_autoencoder):

	# The saved adversarial autoencoder takes x gives us z_hat and x_hat
	# Let's wrap it in an outer model that takes the string inputs and gives us z and x
	# That way we have everything we need for inference and anomaly detection
	
	vectorization_layer = TextVectorization(output_mode='binary')
	vectorization_layer.set_vocabulary(vocabulary)

	input = Input(shape=(1,), dtype='string')
	x = vectorization_layer(input)
	# The lambda layer is necessary to get floats from the vectorization for input in the subsequent layer
	# Because the trained model layer 'enc1' was built for floats as it can take either x or x_hat
	x_float = tf.keras.layers.Lambda(lambda x:tf.cast(x, 'float32'))(x)
	z = adv_autoencoder.get_layer('enc1')(x_float)
	x_hat = adv_autoencoder.get_layer('dec')(z)
	z_hat = adv_autoencoder.get_layer('enc2')(x_hat)
	inference_ganomaly = Model(input, [x_float, z, x_hat, z_hat])
	inference_ganomaly.compile(optimizer='Adam', loss=['binary_crossentropy', encoder_loss, None, None], metrics=[[autoencoder_accuracy, reverse_autoencoder_false_neg_rate], None, None, None])

	return inference_ganomaly

if __name__ == '__main__':

	profiles, depa, depa_dict = load_data(profiles_file, depa_file, depa_dict_file)

	profiles_test = make_partition_list(profiles, test_years_begin, test_years_end)
	depa_test = make_partition_list(depa, test_years_begin, test_years_end)

	profiles_test, depa_test = filter_partition(profiles_test, depa_test, depa_dict, 'test')
	cat_depa_test = [depa_dict[d[0]] for d in depa_test]
	save_data_file(os.path.join(model_dir, 'cat_depa_list.pkl'), cat_depa_test)
	unique_categorized_departements = list(set(cat_depa_test))

	test_ds = text_to_dataset(list_to_text(profiles_test)).batch(batch_size).prefetch(25)
	
	adv_autoencoder = tf.keras.models.load_model(os.path.join(model_dir), custom_objects={'encoder_loss':encoder_loss, 'autoencoder_accuracy':autoencoder_accuracy, 'autoencoder_false_neg_rate':autoencoder_false_neg_rate})
	vocabulary = load_data_file(os.path.join(model_dir, 'vocabulary.pkl'))

	inference_ganomaly = build_inference_ganomaly(vocabulary, adv_autoencoder)

	x_float, z, x_hat, z_hat = [], [], [], []
	for idx, batch in enumerate(tqdm(test_ds)):
		batch_x_float, batch_z, batch_x_hat, batch_z_hat = inference_ganomaly.predict_on_batch(batch)
		x_float.append(batch_x_float)
		z.append(batch_z)
		x_hat.append(batch_x_hat)
		z_hat.append(batch_z_hat)

	x_float = np.vstack(x_float)
	z = np.vstack(z)
	x_hat = np.vstack(x_hat)
	z_hat = np.vstack(z_hat)

	targets = tf.data.Dataset.from_tensor_slices((x_hat, z_hat, x_float, z))
	evaluate_ds = tf.data.Dataset.zip((test_ds.unbatch(), targets)).batch(batch_size).prefetch(25)

	results = inference_ganomaly.evaluate(evaluate_ds, verbose=1)
	joblib.dump(results, os.path.join(model_dir, 'results.joblib'))

	x_hat_dichot = (x_float >= 0.5) * 1
	joblib.dump(x_float, os.path.join(model_dir, 'eval_true.joblib'))
	joblib.dump(x_hat, os.path.join(model_dir, 'eval_predictions.joblib'))
	
	with open(os.path.join(model_dir,'evaluation_results.txt'), mode='w', encoding='cp1252') as file:
		file.write('Evaluation on test set results\n')
		file.write('Predictions for {} classes\n'.format(len(vocabulary)))
		file.write('{} classes reprensented in targets\n'.format(sum(np.amax(x_hat_dichot, axis=0))))
		for metric, result in zip(inference_ganomaly.metrics_names, results):
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
	
		depa_idxes = [idx for idx, element in enumerate(depa_test) if depa_dict[element[0]] == depa]
		depa_encloss = np.take(encoder_losses, depa_idxes)
		
		depa_encloss_forotsu = depa_encloss[depa_encloss < np.percentile(depa_encloss,90)]
		otsu = threshold_otsu(depa_encloss_forotsu)

		otsu_dict[depa] = otsu

	save_data_file(os.path.join(model_dir, 'otsu_thresholds.pkl'), otsu_dict)
