import os
import joblib

from train_ganomaly import (filter_partition, load_data,
                            make_partition_list,
                            save_data_file)
from train_outliers import DepartmentScorer, pse_a, pse_pp

##############
# Parameters #
##############

# Data files
profiles_file = 'data/paper_data/active_meds_list_test.pkl'
depa_file = 'data/paper_data/depa_list_test.pkl'
depa_dict_file = 'data/paper_data/depas.csv'

# Model dir
model_dir = 'experiments/outliers/if'

# Years to use
test_years_begin = 2018 # inclusively
test_years_end = 2018 # inclusively

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
	'Multispécialité':'Specialized pediatrics',
	'Overall':'Overall'
}

###########
# Execute #
###########

if __name__ == '__main__':

	profiles, depa, depa_dict = load_data(profiles_file, depa_file, depa_dict_file)

	profiles_test = make_partition_list(profiles, test_years_begin, test_years_end)
	depa_test = make_partition_list(depa, test_years_begin, test_years_end)
	profiles_test, depa_test = filter_partition(profiles_test, depa_test, depa_dict, 'test')

	pipeline = joblib.load(os.path.join(model_dir, 'outlier_pipeline.joblib'))

	data = [[profile, depa_dict[depa[0]]] for profile, depa in zip(profiles_test, depa_test)]
	save_data_file(os.path.join(model_dir, 'cat_depa_list.pkl'), [d[1] for d in data])

	predictions = pipeline.predict(data)
	scores = pipeline.score_samples(data)
	coords = pipeline[:-1].transform(data)

	anomaly_ratios = {'{}'.format(depa_mapped):DepartmentScorer(depa).anomaly_ratio(pipeline, data) for depa, depa_mapped in depa_string_mapdict.items()}
	print('\n Anomaly ratios')
	[print('{} : {:.5f}'.format(depa, ratio)) for depa, ratio in anomaly_ratios.items()]
	
	save_data_file(os.path.join(model_dir, 'anomaly_ratios.pkl'), anomaly_ratios)
	joblib.dump(predictions, os.path.join(model_dir, 'predictions.joblib'))
	joblib.dump(scores, os.path.join(model_dir, 'scores.joblib'))
	joblib.dump(coords, os.path.join(model_dir, 'coords.joblib'))

	quit()
