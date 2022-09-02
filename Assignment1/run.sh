#!/bin/sh

# For running sweeps
#python -m wandb sweep --project "project_name" sweep.yaml
#python -m wandb agent "saish/project_name/sweep_id"

#python -m wandb sweep --project "Deep-Learning" sweep.yaml
#python -m wandb agent "saish/Deep-Learning/1wyfksde"

python main.py --optimizer nadam --epochs 5 --l_rate 0.001 --batch_size 32 --loss squared_error --activation relu --n_hlayers 5 --hlayer_size 128 --initializer random
