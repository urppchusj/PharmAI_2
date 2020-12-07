import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from train_ganomaly import load_data_file

##############
# Parameters #
##############

model_dir = 'model'

# Only departments in this dict will be processed, the rest will be dropped
depa_string_mapdict = {
	'Ob/gyn':'Ob/gyn', 
	'Adolescent': 'Adolescent medicine',
	'Chirurgie':'Surgery', 
	'Réadaptation':'Long term care', 
	'Oncologie':'Oncology',
	'Néonatologie':'NICU', 
	'Soins intensifs':'PICU',
	'Pédiatrie':'General pediatrics',
	'Psychiatrie':'Psychiatry', 
	'Pouponnière':'Nursery', 
	'Multispécialité':'Specialized pediatrics'
}

#############
# Functions #
#############

def encoder_loss_graphs(depa_string_mapdict, depa_list, encoder_losses):

	encoder_loss_by_depa_dict = {'Department':[], 'Encoder loss':[]}
	
	for depa in depa_string_mapdict.keys():

		depa_idxes = [idx for idx, element in enumerate(depa_list) if element == depa]
		depa_encloss = np.take(encoder_losses, depa_idxes)

		encoder_loss_by_depa_dict['Department'].append(depa_string_mapdict[depa])
		encoder_loss_by_depa_dict['Encoder loss'].append(depa_encloss)

	# Make the encoder loss by depa graph
	encoder_loss_dict = {k:v for k,v in zip(encoder_loss_by_depa_dict['Department'], encoder_loss_by_depa_dict['Encoder loss'])}
	encloss_by_depa_df = pd.concat({k: pd.Series(v) for k, v in encoder_loss_dict.items()})
	encloss_by_depa_df = encloss_by_depa_df.reset_index()
	encloss_by_depa_df.rename(inplace=True, columns={'level_0':'Department','level_1':'sample_index', 0:'Encoder loss'})
		
	sns.set(font_scale=1.5, style="whitegrid")

	sns.catplot(orient='h', y='Department', x='Encoder loss', data=encloss_by_depa_df, kind='box', showfliers=False, aspect=1.4, order=encloss_by_depa_df[['Department', 'Encoder loss']].groupby('Department').agg('median').sort_values(by='Encoder loss').index.values)

	plt.savefig(os.path.join(model_dir, 'encoder_loss_by_depa.png'), dpi=600)

###########
# Execute #
###########

if __name__ == '__main__':

	# TODO Make this more elegant (also in train.py)
	cat_depa_list = load_data_file(os.path.join(model_dir, 'cat_depa_list.pkl'))
	encoder_losses = joblib.load(os.path.join(model_dir, 'encoder_losses.joblib'))
	otsu_thresholds_dict = load_data_file(os.path.join(model_dir, 'otsu_thresholds.pkl'))

	unique_categorized_departements = list(set(cat_depa_list))

	try:
		for k in depa_string_mapdict.keys():
			assert k in unique_categorized_departements
	except:
		print('ERROR: At least one department in the mapping dict is not present in the data. Check data and dict.')
		quit()

	try:
		for k in depa_string_mapdict.keys():
			assert k in otsu_thresholds_dict.keys()
	except:
		print('ERROR: At least one department in the mapping dict is not present in the otsu thresholds. Check data and dict.')
		quit()
	
	encoder_loss_graphs(depa_string_mapdict, cat_depa_list, encoder_losses)
