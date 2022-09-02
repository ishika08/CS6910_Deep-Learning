# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
# export CUDA_VISIBLE_DECVICES=1

# python -m wandb sweep --project "Deep-Learning-CNN" sweep.yaml
# python -m wandb agent "saish/Deep-Learning-CNN/qsd6ilql"

# python -m wandb agent "saish/deep-learning-cnn/17td9zxe"

# python -m wandb agent "saish/deep-learning-cnn/waxgpmmb"

# python -m wandb agent "saish/deep-learning-cnn/artzdakq"

# python -m wandb agent "saish/deep-learning-cnn/1y5tuv1c"

python main.py  --n_classes 10 --epochs 5 --dropout 0.5 --activation relu --denselayer_size 128 --batch_size 64 --train_model True --model_version InceptionResNetV2 --l_rate 0.001 --optimizer Adam --loss categorical_crossentropy
