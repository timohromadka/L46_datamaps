"""
Utility functions for plotting results
"""

import json
import os
import pandas as pd
import torch
import wandb
from dataset import create_data_module
from models import create_model
from notebooks._utils import DictObj


def save_dataframe(df, file_name='test'):
	"""
	Save a dataframe as a csv file.
	"""
	df.to_csv(f'/home/am2770/Github/cancer-low-data/compute_results/csv/{file_name}.csv')
	df.to_excel(f'/home/am2770/Github/cancer-low-data/compute_results/csv/{file_name}.xlsx')

	print(f'DataFrame saved at /home/am2770/Github/cancer-low-data/compute_results/csv/{file_name}.csv')



def filter_runs(runs, return_dataframe=True):
	"""
	- runs is a list sorted descending by .created_at
	"""

	filtered_runs = []
	for run in runs:
		# filter summary runs
		if 'bestmodel_train/balanced_accuracy' not in run.summary._json_dict.keys() \
			or type(run)==str:
			continue
		
		filtered_runs.append(run)
	
	if return_dataframe:
		return transforms_runs_into_dataframe(filtered_runs)
	else:
		return filtered_runs


def transforms_runs_into_dataframe(runs):
	"""
	Returns
	- Dataframe including
		- all configs
		- performance of best model
	"""
	name_list, ids_list, config_list, summary_list = [], [], [], []
	for i, run in enumerate(runs):
		summary_list.append(run.summary._json_dict)

		# .config contains the hyperparameters. We remove special values that start with _.
		config_list.append(
			{k: v for k,v in run.config.items()
			if not k.startswith('_')}
		)

		# .name is the human-readable name of the run.
		name_list.append(run.name)
		ids_list.append(run.id)

	return pd.concat([
			pd.DataFrame({"name": name_list}),
			pd.DataFrame({"id": ids_list}),
			pd.DataFrame(summary_list),
			pd.DataFrame(config_list)
		],
		axis=1
	)


def add_tag_and_update(runs, tag):
	"""
	Append a tag to a given set of experiments. Used for adding an experiment tag.

	- runs: list of wandb.apis.public.Run
	- tag to append
	"""
	for run in runs:
		if tag not in run.tags:
			run.tags.append(tag)
			run.update()


def create_model_and_load_weights(run_name, api=None):
	"""
	Helper function to recreate the model at a given run:
	- downlod the experiment arguments
	- instantiate the model
	- download the model weights
	- load the weights

	Returns
	- model
	- args
	"""
	if api==None:
		api = wandb.Api()

	### Recreate a model using the same argument from the run
	run = api.run(f'andreimargeloiu/low-data/{run_name}')
	args = DictObj(run.config)


	### Instantiate model
	data_module = create_data_module(args)
	model = create_model(args, data_module)
	print(f"\nModel created")


	### Download the pretrained weights
	artifact_name = f'model-{run_name}:v1'
	wandb_artifact_path = f'andreimargeloiu/low-data/{artifact_name}'

	print(f"\nDownloading artifact: {wandb_artifact_path}...")
	artifact = wandb.use_artifact(wandb_artifact_path, type='model')
	artifact_dir = artifact.download()
	print("\nArtifact downloaded")


	### Load the pretrained weights
	print(f"\nLoading pretrained weights into model...")
	model_checkpoint = torch.load(os.path.join(artifact_dir, 'model.ckpt'))
	weights = model_checkpoint['state_dict']

	missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
	if len(missing_keys)!=0 or len(unexpected_keys)!=0:
		if len(missing_keys)!=0:
			print(f"Missing keys: \n")
			print(missing_keys)

		if len(unexpected_keys)!=0:
			print(f"Unexpected keys: \n")
			print(unexpected_keys)
	else:
		print("\nSuccessfully loaded the weights")

	return model, args


def get_best_result_on_validation(df, drop_not_best=True):
	"""
	Given a dataframe with the results of ONE MODEL for each dataset,
		it sorts based on the validation mean balanced accuracy returns the best result
	"""
	df = df.reset_index()

	assert df.model.nunique()==1

	res = df.sort_values(('bestmodel_valid/balanced_accuracy', 'mean'), ascending=False)\
		.groupby(['dataset']).head(1000).reset_index(drop=True)
	
	if drop_not_best:
		res = res.drop_duplicates(subset=['dataset'], keep='first').set_index('dataset').sort_index()

	return res



def download_feature_importance(ids, api):
	"""
	Given a list of run ids, download the feature importances of the best model evaluated
	"""
	ids_for_df = []
	feature_importances_for_df = []
	for id in ids:
		ids_for_df.append(id)

		# download global feature importance for models
		artifact = api.artifact(f'andreimargeloiu/low-data/run-{id}-bestmodel_train_global_feature_importance:v0')
		datadir = artifact.download()

		json_file_path = f"{datadir}/bestmodel_train_global_feature_importance.table.json" 
		with open(json_file_path, 'r') as j:
			contents = json.loads(j.read())

			feature_importances_for_df.append(list(map(lambda x: x[0], contents['data'])))

	return pd.DataFrame({'id': ids_for_df, 'feature_importances': feature_importances_for_df})


def jaccard_similarity_vectors(a, b):
	a = np.asarray(a)
	b = np.asarray(b)
	intersection = np.sum(np.min(np.stack([a, b], axis=0), axis=0))
	union = np.sum(np.max(np.stack([a, b], axis=0), axis=0))
	
	return intersection / union

def jaccard_similarity_sets(a, b):
	intersection = len(a.intersection(b))
	union = len(a.union(b))
	
	return intersection / union