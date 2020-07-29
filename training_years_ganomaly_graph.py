import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

##############
# Parameters #
##############

training_data_dir = 'experiments/training_years_ganomaly'
num_total_folds = 30
num_folds_per_year = 3

# Map the folds to the number of years

fold_years_dict = {n:(math.floor(n/num_folds_per_year))+1 for n in range(num_total_folds)}

#############
# Functions #
#############

def makegraphs(datapath):

	# load the data file
	training_df = pd.read_csv(os.path.join(datapath, 'training.csv'))

	# Make a new column with the number of years of training data
	training_df['Years of training data'] = training_df['fold'].map(fold_years_dict)

	# make the graphs

	all_data_filtered = training_df[['Years of training data', 'A_dec_autoencoder_accuracy', 'val_A_dec_autoencoder_accuracy']].copy()
	all_data_filtered.rename(inplace=True, index=str, columns={'A_dec_autoencoder_accuracy':'Train accuracy', 'val_A_dec_autoencoder_accuracy':'Val accuracy'})
	all_data_filtered.set_index('Years of training data', inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_0':'Years of training data', 'level_1':'Metric', 0:'Result'})
	all_data_graph_df['Years of training data'] = all_data_graph_df['Years of training data'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Years of training data", y="Result", hue="Metric", hue_order=['Train accuracy', 'Val accuracy'], kind="point", data=all_data_graph_df)
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.savefig(os.path.join(datapath, 'experiments_results_acc.png'))
	results = pd.concat([all_data_graph_df.groupby(['Years of training data', 'Metric']).mean(),all_data_graph_df.groupby(['Years of training data', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath, 'experiment_results_acc.csv'))

	all_data_filtered = training_df[['Years of training data', 'A_dec_autoencoder_false_neg_rate', 'val_A_dec_autoencoder_false_neg_rate']].copy()
	all_data_filtered.rename(inplace=True, index=str, columns={'A_dec_autoencoder_false_neg_rate':'Train false negative rate', 'val_A_dec_autoencoder_false_neg_rate':'Val false negative rate'})
	all_data_filtered.set_index('Years of training data', inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_0':'Years of training data', 'level_1':'Metric', 0:'Result'})
	all_data_graph_df['Years of training data'] = all_data_graph_df['Years of training data'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Years of training data", y="Result", hue="Metric", hue_order=['Train false negative rate', 'Val false negative rate'], kind="point", data=all_data_graph_df)
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.savefig(os.path.join(datapath, 'experiments_results_atypical.png'))
	results = pd.concat([all_data_graph_df.groupby(['Years of training data', 'Metric']).mean(),all_data_graph_df.groupby(['Years of training data', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath,'experiment_results_atypical.csv'))

	all_data_filtered_loss = training_df[['Years of training data', 'A_loss', 'val_A_loss']].copy()
	all_data_filtered_loss.rename(inplace=True, index=str, columns={'A_loss':'Train loss', 'val_A_loss':'Val loss'})
	all_data_filtered_loss.set_index('Years of training data', inplace=True)
	all_data_loss_graph_df = all_data_filtered_loss.stack().reset_index()
	all_data_loss_graph_df.rename(inplace=True, index=str, columns={'level_0':'Years of training data', 'level_1':'Metric', 0:'Loss'})
	all_data_loss_graph_df['Years of training data'] = all_data_loss_graph_df['Years of training data'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Years of training data", y="Loss", hue="Metric", hue_order=['Train loss', 'Val loss'], kind="point", data=all_data_loss_graph_df)
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.savefig(os.path.join(datapath, 'experiments_results_loss.png'), dpi=600)
	results = pd.concat([all_data_loss_graph_df.groupby(['Years of training data', 'Metric']).mean(),all_data_loss_graph_df.groupby(['Years of training data', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath, 'experiment_results_loss.csv'))

#############
## EXECUTE ##
#############

if __name__ == '__main__':

	makegraphs(training_data_dir)
