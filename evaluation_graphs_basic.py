import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import umap
from mpl_toolkits import mplot3d

from train_ganomaly import load_data_file

##############
# Parameters #
##############

model_dir = 'experiments/outliers/if'

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

#############
# Functions #
#############

def make_projection_graph(depas, predictions, coords):

	indices = np.random.choice(np.arange(len(depas)), 2000)
	sampled_depa = np.take(depas, indices)
	sampled_predictions = np.take(predictions, indices)
	sampled_coords = np.take(coords, indices, axis=0)
	umap_vectors = umap.UMAP(n_components=3).fit_transform(sampled_coords)

	graph_marker_dict = {1:'o', -1:'^'}
	graph_markers = [graph_marker_dict[i] for i in sampled_predictions.tolist()]
	graph_color_dict = {depa:n for depa, n in zip(list(set(sampled_depa)), range(len(list(set(sampled_depa)))))}
	graph_colors = [graph_color_dict[depa] for depa in sampled_depa]

	graph_data = []
	for umapcoords, color, marker in zip(umap_vectors, graph_colors, graph_markers):
			graph_data.append([umapcoords[0], umapcoords[1], umapcoords[2], color, marker])

	graph_data_df = pd.DataFrame(data=graph_data, columns=['x', 'y', 'z', 'color', 'marker'])

	ax = plt.figure(figsize=(16,10)).gca(projection='3d')
	for marker, group in graph_data_df.groupby('marker'):
			ax.scatter(
					xs=group['x'] ,
					ys=group['y'] ,
					zs=group['z'],
					c=group['color'],
					marker=marker,
					cmap='prism'
			)
	plt.savefig(os.path.join(model_dir, 'umap_projection.png'))
	plt.gcf().clear()

def make_score_graph(depas, scores):

	graph_data = []
	for score, depa in zip(scores, depas):
		if depa in depa_string_mapdict.keys():	
			graph_data.append([score, depa_string_mapdict[depa]])
	graph_data_df = pd.DataFrame(data=graph_data, columns=['Anomaly score', 'Department'])

	sns.set(font_scale=1.5, style="whitegrid")

	sns.catplot(orient='h', y='Department', x='Anomaly score', data=graph_data_df, kind='box', showfliers=False, aspect=1.4, order=graph_data_df.groupby('Department').agg('median').sort_values(by='Anomaly score').index.values)

	plt.savefig(os.path.join(model_dir, 'anomaly_score_by_depa.png'), dpi=600)
	plt.gcf().clear()

def make_ratio_graph(ratios):

	ratio_df = pd.DataFrame.from_dict(ratios, orient='index', columns=['Anomaly ratio'])
	ratio_df.reset_index(inplace=True)
	ratio_df.rename(inplace=True, index=str, columns={'index':'Department'})
	
	sns.set(font_scale=1.5, style="whitegrid")
	sns.barplot(orient='h', data=ratio_df, y='Department', x='Anomaly ratio', order=ratio_df.sort_values(by='Anomaly ratio')['Department'])
	plt.subplots_adjust(left=0.45, bottom=0.15)

	plt.savefig(os.path.join(model_dir, 'anomaly_ratio_by_depa.png'), dpi=600)
	plt.gcf().clear()

###########
# Execute #
###########

if __name__ == '__main__':

	cat_depa_list = load_data_file(os.path.join(model_dir, 'cat_depa_list.pkl'))
	anomaly_ratios = load_data_file(os.path.join(model_dir, 'anomaly_ratios.pkl'))
	coords = joblib.load(os.path.join(model_dir, 'coords.joblib'))
	predictions = joblib.load(os.path.join(model_dir, 'predictions.joblib'))
	scores = joblib.load(os.path.join(model_dir, 'scores.joblib'))

	make_projection_graph(cat_depa_list, predictions, coords)
	make_score_graph(cat_depa_list, scores)
	make_ratio_graph(anomaly_ratios)
