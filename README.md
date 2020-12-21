# PixL2R


This is the code for our CoRL 2020 paper [PixL2R: Guiding Reinforcement Learning Using Natural Language by Mapping Pixels to Rewards](https://arxiv.org/abs/2007.15543) 

## Setup:
1) Create and activate conda environment using the `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate pixl2r
```
2) Install dependencies for metaworld, by running the following command in `metaworld/` directory:
```bash
pip install -e .
```
3) Download and extract preprocessed data, by running the following commands in `PixL2R` directory:
```bash
wget https://www.cs.utexas.edu/users/ml/PixL2R/data.zip
unzip data.zip -d data/
```
(The downloaded file is ~3GB; the extracted files require 50GB of disk space.)


## Running Experiments

1. **Supervised learning of PixL2R model**: Run the following command in `src/supervised`
```bash
python model.py --save-path=<save-path>
```
2. **Policy training without language-based rewards**: Run the following command in `src/rl`
```bash
python train_policy.py --obj-id=6 --env-id=1 --reward-type=<sparse|dense>
```
3. **Policy training with language-based rewards**: Pass the PixL2R model filepath (output of step 1) and the description id to use.
```bash
python train_policy.py --obj-id=6 --env-id=1 --reward-type=<sparse|dense> --model-file=/path/to/PixL2R/model --descr-id=<0|1|2>
```


## Data:
The raw videos can be downloaded from the following link: https://www.cs.utexas.edu/users/ml/PixL2R/videos.zip  
The videos were generated using the script `src/rl/generate_videos.py`.


## Acknowledgements
The codebase is based on the following repositories:
1. https://github.com/rlworkgroup/metaworld
2. https://github.com/nikhilbarhate99/PPO-PyTorch
