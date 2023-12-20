import wandb

from pytorch_lightning.loggers import WandbLogger

# def get_run_name(args):
# 	if args.model=='dnn':
# 		run_name = 'mlp'
# 	elif args.model=='dietdnn':
# 		run_name = 'mlp_wpn'
# 	else:
# 		run_name = args.model

# 	if args.sparsity_type=='global':
# 		run_name += '_SPN_global'
# 	elif args.sparsity_type=='local':
# 		run_name += '_SPN_local'

# 	return run_name

def create_wandb_logger(args, project_name):
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