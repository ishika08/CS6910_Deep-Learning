# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
# export CUDA_VISIBLE_DEVICES=0

# python -m wandb sweep --project "Deep-Learning-CNN" sweep.yaml
# python -m wandb agent "saish/Deep-Learning-CNN/p23ec4hq"

# python -m wandb agent "saish/Deep-Learning-CNN/yxc70guc"

# python -m wandb agent "saish/Deep-Learning-CNN/xnkyukvj"

python main.py --n_filters 32 --filter_multiplier 1 --epochs 10 --activation relu --batch_size 32 --batch_norm No --train_model True --optimizer Adam --l_rate 0.001 --dropout 0.5 --denselayer_size 128 --filter_size 3 --loss categorical_crossentropy
