import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

##############
# Parameters #
##############

training_data_dir = 'experiments/training_years_outliers'
n_folds = 3
depas_to_score = ['Overall', 'Néonatologie', 'Ob/gyn', 'Oncologie', 'Pédiatrie']

anomaly_algorithm_rename_dict = {'LocalOutlierFactor(contamination=0.2, novelty=True)':'LOF', 'EllipticEnvelope(contamination=0.2)':'EE', 'IsolationForest(contamination=0.2)':'IF', 'OneClassSVM(nu=0.2)':'OC SVM'}

x_axis_name_in_figure = 'Years'
algorith_name_in_figure = 'Alg'
n_components_name_in_figure = 'components'
column_name_rename_dict = {'param_tsvd__n_components':n_components_name_in_figure, 'param_anomaly_algorithm':algorith_name_in_figure}

columns_to_extract = ['split{}_test_Ratio anomalies {}'.format(n, depa) for depa in depas_to_score for n in range(n_folds)]
columns_to_extract.extend([x_axis_name_in_figure, 'param_tsvd__n_components', 'param_anomaly_algorithm'])

metric_rename_dict = {'split{}_test_Ratio anomalies {}'.format(n,depa):'Ratio anomalies {}'.format(depa) for depa in depas_to_score for n in range(n_folds)}

#############
# Functions #
#############

def makegraphs(datapath):

	# load the data files and concatenate them into a single pandas dataframe
	files_data = []
	for file in os.listdir(datapath):
		if file.endswith('.csv'):
			file_df = pd.read_csv(os.path.join(datapath, file))
			file_df[x_axis_name_in_figure] = os.path.splitext(file)[0]
			files_data.append(file_df)
	all_data = pd.concat(files_data)

	all_data_filtered = all_data[columns_to_extract]

	all_data_filtered.rename(inplace=True, index=str, columns=column_name_rename_dict)
	all_data_filtered[algorith_name_in_figure] = all_data_filtered[algorith_name_in_figure].map(anomaly_algorithm_rename_dict)
	all_data_filtered.set_index([x_axis_name_in_figure, algorith_name_in_figure, n_components_name_in_figure], inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_3':'Metric', 0:'Result'})
	all_data_graph_df['Metric'] = all_data_graph_df['Metric'].map(metric_rename_dict)
	all_data_graph_df[x_axis_name_in_figure] = all_data_graph_df[x_axis_name_in_figure].astype('int8')
	sns.set(style="whitegrid", font_scale=2)
	sns.catplot(font_scale = 2, x=x_axis_name_in_figure, y="Result", hue="Metric", row=algorith_name_in_figure, col=n_components_name_in_figure,data=all_data_graph_df, kind='point', margin_titles=True)
	plt.savefig(os.path.join(datapath, 'training_years_results.png'))


#############
## EXECUTE ##
#############

if __name__ == '__main__':

	makegraphs(training_data_dir)


