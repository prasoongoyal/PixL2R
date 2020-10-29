# PixL2R


This is the code for our CoRL 2020 paper [PixL2R: Guiding Reinforcement Learning Using Natural Language by Mapping Pixels to Rewards](https://arxiv.org/abs/2007.15543) 

## Setup:
1) Create conda environment using the environment.yml file.

## Generate videos to annotate: 
1) Update the metaworld root path in src/rl/generate_videos.py
2) Run the following command in src/rl:
```bash
python generate_videos.py --obj-id=6 --start=0 --end=100
```

## Supervised learning:
1) Set up data.
2) Run the following command in src/supervised
```bash
python model.py --save-path=<save-path>
```

## Policy training: 
1) Update the metaworld root path in src/rl/train_policy.py
2) Set CUDA_VISIBLE_DEVICES environment variable to the desired GPU.
3) Run the following command in src/rl to train the model with only extrinsic sparse or dense rewards:
```bash
python train_policy.py --obj-id=6 --env-id=1 --reward-type=<sparse|dense>
```
To train the model with language-based rewards in addition to extrinsic rewards, pass the PixL2R model file and the description id to use:
```bash
python train_policy.py --obj-id=6 --env-id=1 --reward-type=<sparse|dense> --model-file=/path/to/PixL2R/model --descr-id=<0|1|2>
```

## Acknowledgements
The codebase is based on the following repositories:
1. https://github.com/rlworkgroup/metaworld
2. https://github.com/nikhilbarhate99/PPO-PyTorch
