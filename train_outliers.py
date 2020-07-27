import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import OneClassSVM

from train_ganomaly import (execution_checks, filter_partition, load_data,
                            load_data_file, make_partition_list,
                            save_data_file)

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
train_years_begin = [2014,2015,2016] # inclusively
train_years_end = [2014,2015,2016]# inclusively
val_years_begin = [2015,2016,2017] # inclusively
val_years_end = [2015,2016,2017] # inclusively

# Model parameters
depas_to_score = ['Overall', 'Néonatologie', 'Ob/gyn', 'Oncologie', 'Pédiatrie']
contamination_ratio = 0.2
param_grid = dict(
				tsvd__n_components = [1024],
				anomaly_algorithm = [LocalOutlierFactor(novelty=True, contamination=contamination_ratio), IsolationForest(contamination=contamination_ratio), OneClassSVM(nu=contamination_ratio, gamma='scale'), OneClassSVM(nu=contamination_ratio, gamma='auto')], #EllipticEnvelope(contamination=contamination_ratio), 
				)
tsvd_n_components = 64

###########
# Classes #
###########   

class YearsSplitter():

	def __init__(self, years_indices, train_years_begin, train_years_end, val_years_begin, val_years_end):

		self.years_indices = np.array(years_indices)
		self.train_years_begin = train_years_begin
		self.train_years_end = train_years_end
		self.val_years_begin = val_years_begin
		self.val_years_end = val_years_end

	def split(self, X):

		for train_year_begin, train_year_end, val_year_begin, val_year_end in zip(train_years_begin, train_years_end, val_years_begin, val_years_end):
		
			indices = np.arange(len(X))
			train_indices = np.squeeze(np.take(indices,np.argwhere(np.isin(self.years_indices, np.arange(train_year_begin, train_year_end + 1)))))
			val_indices = np.squeeze(np.take(indices,np.argwhere(np.isin(self.years_indices, np.arange(val_year_begin, val_year_end + 1)))))

			yield train_indices, val_indices

class DepartmentScorer:

	def __init__(self, depa):
		self.depa = depa

	def anomaly_ratio(self, estimator, X):
		if self.depa=='Overall':
			predictions = estimator.predict(X)
		else:
			predictions = estimator.predict([x for x in X if x[1] == self.depa])
		n_predictions = len(predictions)
		n_anomalies = np.sum(predictions == -1)
		return n_anomalies / n_predictions

#############
# Functions #
#############

# string preprocessor (join the strings with spaces to simulate a text)
def pse_pp(x):
	return ' '.join(x)

# string analyzer (do not transform the strings, use them as is because they are not words.)
def pse_a(x):
	return x

###########
# Execute #
###########

if __name__ == '__main__':

	# Check that the provided years are okay and define execution mode

	validate, _, _, _, _, _ = execution_checks(save_dir, train_years_begin, train_years_end, val_years_begin, val_years_end)

	# Load data

	profiles, depa, depa_dict = load_data(profiles_file, depa_file, depa_dict_file)

	# Prepare the data

	all_years_range = np.unique(np.hstack([np.arange(beg, end + 1) for beg, end in zip(train_years_begin, train_years_end)]))
	if validate:
		val_years_range = np.unique(np.hstack([np.arange(beg, end + 1) for beg, end in zip(val_years_begin, val_years_end)]))
		all_years_range = np.unique(np.hstack([all_years_range, val_years_range]))
	
	filtered_profiles = []
	filtered_depa = []
	years_indices = []
	
	for year in all_years_range:

		profiles_year = make_partition_list(profiles, year, year)
		depa_year = make_partition_list(depa, year, year)
		profiles_year, depa_year = filter_partition(profiles_year, depa_year, depa_dict, year)
		
		filtered_profiles.extend(profiles_year)
		filtered_depa.extend(depa_year)
		years_indices.extend(np.repeat(year, len(profiles_year)))
	
	# Train

	model = Pipeline([
		('columntrans', ColumnTransformer([
				('profiles', CountVectorizer(lowercase=False, preprocessor=pse_pp, analyzer=pse_a), 0),
				('depas', 'drop', 1),
		])),
		('tfidf', TfidfTransformer()),
		('tsvd', TruncatedSVD(n_components=tsvd_n_components)),
		('anomaly_algorithm', IsolationForest())
	], verbose=True)

	data = [[profile, depa_dict[depa[0]]] for profile, depa in zip(filtered_profiles, filtered_depa)]

	if validate:

		splitter = YearsSplitter(years_indices, train_years_begin, train_years_end, val_years_begin, val_years_end)
		score_dict = {'Ratio anomalies {}'.format(depa):DepartmentScorer(depa).anomaly_ratio for depa in depas_to_score}
		search = GridSearchCV(model, param_grid=param_grid, cv=splitter.split(filtered_profiles), scoring=score_dict, verbose=True, n_jobs=-2, refit=False)

		search.fit(data)

		os.makedirs(save_dir)
		results = pd.DataFrame.from_dict(search.cv_results_)
		results.to_csv(os.path.join(save_dir, 'cv_results.csv'))
	
	else:

		model.fit(depa)
		os.makedirs(save_dir)
		joblib.dump(model, os.path.join(save_dir, 'outlier_pipeline.joblib'))
