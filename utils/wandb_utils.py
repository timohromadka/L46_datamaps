import json
import wandb

import pandas as pd
from pytorch_lightning.loggers import WandbLogger

def create_wandb_logger(args, project_name):
	wandb.finish()
	wandb_logger = WandbLogger(
		project=project_name,
		group=args.group,
		job_type=args.job_type,
		tags=args.tags,
		notes=args.notes,
		# reinit=True,

		log_model=args.wandb_log_model,
		settings=wandb.Settings(start_method="thread")
	)
	wandb_logger.experiment.config.update(args)	  # add configuration file

	return wandb_logger


def flatten_config(config):
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # For nested dictionaries, you can either flatten further or handle specific cases
            # Here, assuming a shallow nested structure
            for sub_key, sub_value in value.items():
                flat_config[f"{key}_{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    return flat_config

def get_metrics(run, metric_names):
    metrics = {}
    for name in metric_names:
        # Fetching the metric's summary might vary depending on how it's logged
        # Using '.get()' to avoid KeyError if the metric is not found
        metrics[name] = run.summary.get(name)
    return metrics

def get_runs_dataframe(project_name, entity_name='l46_datamaps'):
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")
        
    metric_names = ["test_recall", "test_loss", "val_loss", "test_precision", "test_f1", "test_acc", "val_loss", "val_acc", "epoch", "train_acc_epoch", "train_loss_epoch", "lr-Adam"]
    runs_data = []
    for run in runs:
        run_data = {
            'id': run.id,
            'name': run.name,
            'state': run.state,
        }
        
        run_data.update(flatten_config(run.config))
        run_data.update(get_metrics(run, metric_names))
        runs_data.append(run_data)

    return pd.DataFrame(runs_data)

def get_run_id_from_run_name(run_name, run_df):
    return run_df.loc[run_df['wandb_run_name'] == run_name, 'id'].iloc[0]

def get_training_dynamics(api, project_name, entity_name, run_id, run_name):
    run = api.run(f"{entity_name}/{project_name}/{run_id}")
    json_file_name = f'training_dynamics/{run_name}_training_dynamics.json'
    file_object = run.file(json_file_name).download(replace=True)
    training_dynamics = json.load(file_object)
    
    zipped_values = zip(*[(values['gold_label_probs'], values['confidence'], values['variability'], values['correctness'], values['forgetfulness'])
                      for values in training_dynamics.values()])

    return zipped_values

def get_training_dynamics_from_run_name(project_name, entity_name, run_name):
    run_df = get_runs_dataframe(project_name, entity_name)
    run_id = get_run_id_from_run_name(run_name, run_df)
    api = wandb.Api()
    training_dynamics = get_training_dynamics(api, project_name, entity_name, run_id, run_name)
    
    return training_dynamics

def get_val_split_seed_from_run_name(teacher_run_name, project_name, entity_name):
    run_df = get_runs_dataframe(project_name, entity_name)
    matching_run = run_df.loc[run_df['name'] == teacher_run_name]

    # no match found on teacher_run_name
    if matching_run.empty:
        raise ValueError(f"No run found with name {teacher_run_name}")

    val_split_seed = matching_run['val_split_seed'].iloc[0]

    # make sure that the val_split_seed is definitely an INTEGER, albeit in float format
    if not (isinstance(val_split_seed, int) or isinstance(val_split_seed, float) and val_split_seed.is_integer): 
        raise ValueError(f"Value of val_split_seed '{val_split_seed}' is not a valid integer")

    return int(val_split_seed)