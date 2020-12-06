import argparse as ap
import os
import pathlib
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm


class preprocessor():

	def __init__(self, source_file):
		
		# Load raw data
		print('Loading data...')
		self.raw_profile_data = pd.read_csv(source_file, sep=';', parse_dates=['date_begenc', 'datetime_beg', 'datetime_end', 'datetime_endenc'])
		self.raw_profile_data.sort_values(['date_begenc', 'enc', 'datetime_beg'], ascending=True, inplace=True)
		self.raw_profile_data['addition_number'] = self.raw_profile_data.groupby('enc').enc.rank(method='first').astype(int)
		self.raw_profile_data.set_index(['enc', 'addition_number'], drop=True, inplace=True)
		self.raw_profile_data['year'] = self.raw_profile_data['date_begenc'].apply(lambda x: x.year)

	def get_profiles(self):
		# Rebuild profiles at every addition
		print('Recreating profiles... (takes a while)')
		profiles_dict = defaultdict(list)
		active_profiles_dict = defaultdict(list)
		depa_dict = defaultdict(list)
		enc_list = []
		# Iterate over encounters, send each encounter to self.build_enc_profiles
		for enc in tqdm(self.raw_profile_data.groupby(level='enc', sort=False)):
			enc_list.append(enc[0])
			profiles_dict[enc[1]['year'].iloc[0]].append(enc[1]['medinb_int'].tolist())
			enc_profiles = self.build_enc_profiles(enc)
			# Convert each profile to list
			for profile in enc_profiles.groupby(level='profile', sort=False):
				active_profile_to_append_list, depa_to_append_list = self.make_profile_lists(profile)
				key = enc[1]['year'].iloc[0]
				active_profiles_dict[key].extend(active_profile_to_append_list)
				depa_dict[key].extend(depa_to_append_list)
		return profiles_dict, active_profiles_dict, depa_dict, enc_list

	def build_enc_profiles(self, enc):
		enc_profiles_list = []
		prev_add_time = pd.Timestamp(enc[1]['datetime_beg'].values[0])
		max_enc = enc[1].index.get_level_values('addition_number').max()
		# Iterate over additions in the encounter
		for addition in enc[1].itertuples():
			# For each addition, generate a profile of all medications with a datetime of beginning
			# before or at the same time of the addition
			# Generate profiles only when no drug was added for 1 hour, representing a "stable" profile for retrospective analysis of all drugs in the profile
			cur_add_time = addition.datetime_beg
			if addition.Index[1] == max_enc:
				pass
			elif cur_add_time < prev_add_time + pd.DateOffset(hours=1):
				continue
			profile_at_time = enc[1].loc[(enc[1]['datetime_beg'] <= addition.datetime_beg)].copy()
			# Determine if each medication was active at the time of addition
			profile_at_time['active'] = np.where(profile_at_time['datetime_end'] > addition.datetime_beg, 1, 0)
			# Manipulate indexes to have three levels: encounter, profile and addition
			profile_at_time['profile'] = addition.Index[1]
			profile_at_time.set_index('profile', inplace=True, append=True)
			profile_at_time = profile_at_time.swaplevel(i='profile', j='addition_number')
			enc_profiles_list.append(profile_at_time)
			# Calculate how much time elapsed since last addition
			prev_add_time = cur_add_time
		enc_profiles = pd.concat(enc_profiles_list)
		return enc_profiles

	def make_profile_lists(self, profile):
		active_profile_to_append_list = []
		depa_to_append_list = []
		active_profile = profile[1].loc[profile[1]['active'] == 1].copy()
		# make lists of contents of active profile to prepare for multi-hot encoding
		active_profile_to_append = active_profile['medinb_int'].tolist()
		depa_to_append = active_profile['cat_depa'].unique().tolist()
		active_profile_to_append_list.append(active_profile_to_append)
		depa_to_append_list.append(depa_to_append)
		return active_profile_to_append_list, depa_to_append_list

	def preprocess(self):
		# Preprocess the data
		profiles_dict, active_profiles_dict, depa_dict, enc_list = self.get_profiles()
		# Save preprocessed data to pickle file
		data_save_path = os.path.join('data/paper_data')
		pathlib.Path(data_save_path).mkdir(parents=True, exist_ok=True)
		with open(os.path.join(data_save_path, 'profiles_list.pkl'), mode='wb') as file:
			pickle.dump(profiles_dict, file)
		with open(os.path.join(data_save_path, 'active_meds_list.pkl'), mode='wb') as file:
			pickle.dump(active_profiles_dict, file)
		with open(os.path.join(data_save_path, 'depa_list.pkl'), mode='wb') as file:
			pickle.dump(depa_dict, file)
		with open(os.path.join(data_save_path, 'enc_list.pkl'), mode='wb') as file:
			pickle.dump(enc_list, file)

###########
# EXECUTE #
###########

if __name__ == '__main__':
	parser = ap.ArgumentParser(description='Preprocess the data extracted from the pharmacy database before input into the machine learning model', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--sourcefile', metavar='Type_String', type=str, nargs="?", default='data/anonymized_data/train_val_set.csv', help='Source file load. Defaults to "data/anonymized_data/train_val_set.csv".')
	
	args = parser.parse_args()
	source_file = args.sourcefile

	try:
		if(not os.path.isfile(source_file)):
			print(
				'Data file: {} not found. Quitting...'.format(source_file))
			quit()
	except TypeError:
		print('Invalid data file given. Quitting...')
		quit()

	pp = preprocessor(source_file)
	pp.preprocess()
