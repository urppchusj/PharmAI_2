import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

##############
# Parameters #
##############

data_dir = 'experiments/outliers_variance'
contamination_ratio = 0.2
n_folds = 3
depas_to_score = ['Overall', 'Néonatologie', 'Ob/gyn', 'Oncologie', 'Pédiatrie']

anomaly_algorithm_rename_dict = {'LocalOutlierFactor(contamination={}, novelty=True)'.format(contamination_ratio):'LOF', 'IsolationForest(contamination={})'.format(contamination_ratio):'IF', 'OneClassSVM(nu={})'.format(contamination_ratio):'OC SVM', 'EllipticEnvelope(contamination={}'.format(contamination_ratio):'EE'}

x_axis_name_in_figure = 'Components'
algorith_name_in_figure = 'Alg'
column_name_rename_dict = {'param_tsvd__n_components':x_axis_name_in_figure, 'param_anomaly_algorithm':algorith_name_in_figure}

columns_to_extract_depascore = ['split{}_test_Ratio anomalies {}'.format(n, depa) for depa in depas_to_score for n in range(n_folds)]
columns_to_extract_variance = ['split{}_test_explained_variance'.format(n) for n in range(n_folds)]
columns_to_extract_depascore.extend([x_axis_name_in_figure, algorith_name_in_figure])
columns_to_extract_variance.extend([x_axis_name_in_figure, algorith_name_in_figure])

metric_rename_dict = {'split{}_test_Ratio anomalies {}'.format(n,depa):'Ratio anomalies {}'.format(depa) for depa in depas_to_score for n in range(n_folds)}

#############
# Functions #
#############

def make_parameter_graphs(df):

	df = df[columns_to_extract_depascore]
	df.set_index([x_axis_name_in_figure, algorith_name_in_figure], inplace=True)
	graph_df = df.stack().reset_index()
	graph_df.rename(inplace=True, index=str, columns={'level_2':'Metric', 0:'Result'})
	graph_df['Metric'] = graph_df['Metric'].map(metric_rename_dict)
	#graph_df[x_axis_name_in_figure] = graph_df[x_axis_name_in_figure].astype('int8')
	sns.set(style="whitegrid", font_scale=2)
	f = sns.catplot(font_scale = 2, x=x_axis_name_in_figure, y="Result", hue="Metric", col=algorith_name_in_figure,data=graph_df, kind='point', margin_titles=True)
	f.set_xticklabels(rotation=35,  horizontalalignment='right')
	plt.savefig(os.path.join(data_dir, 'parameter_results.png'))

def make_variance_graphs(df):

	df = df[columns_to_extract_variance]
	df.set_index([x_axis_name_in_figure, algorith_name_in_figure], inplace=True)
	graph_df = df.stack().reset_index()
	graph_df.rename(inplace=True, index=str, columns={0:'Explained variance ratio'})
	#graph_df[x_axis_name_in_figure] = graph_df[x_axis_name_in_figure].astype('int8')
	sns.set(style="whitegrid", font_scale=2)
	f = sns.catplot(font_scale = 2, x=x_axis_name_in_figure, y="Explained variance ratio", data=graph_df, kind='point')
	f.set_xticklabels(rotation=35,  horizontalalignment='right')
	plt.ylim(0,1)
	plt.savefig(os.path.join(data_dir, 'variance_ratio_results.png'))


#############
## EXECUTE ##
#############

if __name__ == '__main__':

	# load the data files and concatenate them into a single pandas dataframe
	files_data = []
	for file in os.listdir(data_dir):
		if file.endswith('.csv'):
			file_df = pd.read_csv(os.path.join(data_dir, file))
			files_data.append(file_df)
	all_data = pd.concat(files_data)

	all_data.rename(inplace=True, index=str, columns=column_name_rename_dict)
	all_data[algorith_name_in_figure] = all_data[algorith_name_in_figure].map(anomaly_algorithm_rename_dict)

	make_parameter_graphs(all_data)
	make_variance_graphs(all_data)

