import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

##############
# Parameters #
##############

data_dir = 'experiments/outliers_parameters'
n_folds = 3
depas_to_score = ['Overall', 'Néonatologie', 'Ob/gyn', 'Oncologie', 'Pédiatrie']

anomaly_algorithm_rename_dict = {'IsolationForest(contamination=0.2)':'IF', 'OneClassSVM(nu=0.2)':'OC SVM'}

x_axis_name_in_figure = 'Components'
algorith_name_in_figure = 'Alg'
column_name_rename_dict = {'param_tsvd__n_components':x_axis_name_in_figure, 'param_anomaly_algorithm':algorith_name_in_figure}

columns_to_extract = ['split{}_test_Ratio anomalies {}'.format(n, depa) for depa in depas_to_score for n in range(n_folds)]
columns_to_extract.extend(['param_tsvd__n_components', 'param_anomaly_algorithm'])

metric_rename_dict = {'split{}_test_Ratio anomalies {}'.format(n,depa):'Ratio anomalies {}'.format(depa) for depa in depas_to_score for n in range(n_folds)}

#############
# Functions #
#############

def makegraphs(datapath):

	# load the data file
	all_data = pd.read_csv(os.path.join(datapath, 'cv_results.csv'))

	all_data_filtered = all_data[columns_to_extract]

	all_data_filtered.rename(inplace=True, index=str, columns=column_name_rename_dict)
	all_data_filtered[algorith_name_in_figure] = all_data_filtered[algorith_name_in_figure].map(anomaly_algorithm_rename_dict)
	all_data_filtered.set_index([x_axis_name_in_figure, algorith_name_in_figure], inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_2':'Metric', 0:'Result'})
	all_data_graph_df['Metric'] = all_data_graph_df['Metric'].map(metric_rename_dict)
	#all_data_graph_df[x_axis_name_in_figure] = all_data_graph_df[x_axis_name_in_figure].astype('int8')
	sns.set(style="whitegrid", font_scale=2)
	sns.catplot(font_scale = 2, x=x_axis_name_in_figure, y="Result", hue="Metric", col=algorith_name_in_figure,data=all_data_graph_df, kind='point', margin_titles=True)
	plt.savefig(os.path.join(datapath, 'parameter_results.png'))


#############
## EXECUTE ##
#############

if __name__ == '__main__':

	makegraphs(data_dir)


