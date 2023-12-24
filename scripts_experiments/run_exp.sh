python3 run_experiment.py \
    --model resnet \
    --model_size small \
    --dataset cifar10 \
    --epochs 2 \
    --track_training_dynamics \
    --prev_run_name_for_dynamics run_20231223_161559 \
    --p_variability 0.33 \
	--tags 'bad' 'datamap dataloader debug' \
    --notes 'second debug run for datamap dataloader'   