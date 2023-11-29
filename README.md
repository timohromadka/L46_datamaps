# WPFS - Weight Predictor Network with Feature Selection

## Installing the project 
You must have **conda** installed locally. All project dependencies all included in the conda file: environment.yml.

This was tested to work on: CUDA Version: 11.2 (NVIDIA-SMI 460.32.03).

```
<!-- Install the codebase -->
cd REPOSITORY
conda create python=3.7.9 --name low-data
conda activate low-data
pip install -r requirements.txt

<!-- Optionally, install lightgbm -->
# pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```

Change `BASE_DIR` from `/src/_config.py` to point to the project directory on your machine.

Search `am2770`in the codebase and replace the paths with the paths on your machine.

When you install new modules, save the environment:
```
pip freeze > requirements.txt
```

## Codebase basics
- Store training metrics using **wandb**. It makes running experiments much easier.
- Train using **pytorch-lightening**, which is a wrapper on top of pytorch. It has three main benefits: (1) it make training logic much simpler, (2) it allows us to use the same code for training on CPU or GPU, and (3) integrates nicely with wandb.

## Code structure
- src
	- `run_experiment.py`: code for parsing arguments, and starting experiment
		- def parse_arguments - include all command line arguments
		- def run_experiment - create wandb logger and start training model
		- def train_model - training neural networks using pytorch lightning
		- important commandline arguments
			- dataset
			- model
			- feature_extractor_dims - the size of the hidden layers in the dnn
			- max_steps - maximum training iterations
			- batchnorm, dropout_rate
			- lr, batch_size, patience_early_stopping
			- lr_scheduler - learning rate scheduler (the default scheduler reduces the learning rate until it gets 10x smaller)


	- `dataset.py`: loading the datasets
	- `model.py`: neural network architecture
		- def create_model - logic to create a neural network
		- class TrainingLightningModule - generic class that includes logging losses etc using pytorch-lightning
		- class DNN - neural network architecture. It's a very extensible class, that includes more things that you'll need.
		- class FirstLinearLayer - used for adding different first layers (e.g., DietNetworks, ConcreteLayer etc)
		- class WeightPredictorNetwork - for DietNetworks
- scripts_experiments
	- template to set a wandb sweep
- compute_results
	- template to retrieve results from wandb (using SQL-style notation), and analyse them easily
- some utilities
	- run_exp_test.sh - run a test experiment
	- generate_uuid.py - generate a unique id for an experiment
	- kill_tmux_session.sh $1 - kill a specific tmux session


## Changing to a specific branch
```
git pull origin BRANCH
git checkout BRANCH
```
	

# WandB sweep
- note: the wandb sweep run on the current codebase. Thus, if you change the code while running the sweep, new runs will run on the new codebase.
```
wandb sweep scripts_experiments/_template_sweep.yaml
wandb agent XYZ
```

# Tmux sessions
- Tmux is a way to run multiple processes in the same terminal. It's useful for running multiple experiments at the same time. It allows your connection to persist even if you close your terminal.
```
tmux # start a new tmux session
tmux ls # list all tmux sessions
tmux attach -t 0 # attach to a specific tmux session
tmux kill-session -t 0 # kill a specific tmux session
```

Usually you can start multiple runs simultaneously using tmux, to fully use the CPU/GPU. For example, you can run 4 runs in parallel using 4 tmux sessions. E.g., start one new `wandb agent` within each tmux session. You can then kill all tmux sessions using `kill_tmux_session.sh $1` where $1 is the number of tmux sessions you want to kill.