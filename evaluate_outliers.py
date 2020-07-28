import os

import joblib
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from train_ganomaly import (execution_checks, filter_partition, load_data,
                            load_data_file, make_partition_list,
                            save_data_file)

from train_outliers import DepartmentScorer, pse_a, pse_pp

# requires datashader holowviews bokeh

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

# Model parameters
depas_to_score = ['Overall', 'Néonatologie', 'Ob/gyn', 'Oncologie', 'Pédiatrie']

###########
# Execute #
###########

if __name__ == '__main__':

	profiles, depa, depa_dict = load_data(profiles_file, depa_file, depa_dict_file)

	profiles_test = make_partition_list(profiles, test_years_begin, test_years_end)
	depa_test = make_partition_list(depa, test_years_begin, test_years_end)
	profiles_test, depa_test = filter_partition(profiles_test, depa_test, depa_dict, 'test')
	cat_depa_test = [depa_dict[d[0]] for d in depa_test]

	pipeline = joblib.load(os.path.join(model_dir, 'outlier_pipeline.joblib'))

	data = [[profile, depa_dict[depa[0]]] for profile, depa in zip(profiles_test, depa_test)]

	predictions = pipeline.predict(data)
	scores = pipeline.decision_function(data)
	coords = pipeline[:-1].transform(data)

	indices = np.random.choice(np.arange(len(data)), 2000)
	sampled_depa = np.take(cat_depa_test, indices)
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
			zs=group['z'] ,
			c=group['color'],
			marker=marker,
			cmap='prism'
		)
	plt.savefig(os.path.join(model_dir, 'umap_projection.png'))

	quit()

