# Just Flip: Flipped Observation Generation and Optimization for Neural Radiance Fields to Cover Unobserved View

Author : Minjae Lee, Kyeongsu Kang and Hyeonwoo Yu

<br/>

## Overview
<img src="figs/overview.jpeg"  width="800" height="250">
(Left) the baseline approach where the robot only observes one side of an object while driving. This case does not yield good rendering results in unobserved views that the robot has not explored. (Right) our method generates the flipped observations from the actual observations. The robot exploits both input images and flipped images and estimated camera poses to learn 3D space using NeRF for unexplored regions as well. Our method obtains qualified rendering results in unobserved views, even without providing images from unobserved views as a training set.

<br/>

## Dataset
We used NeRF synthethic dataset. Download it through the official link [here]. \
Or you can use your own dataset.

[here]: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

<br/>

## Enviroment setup

Since our baseline is gnerf, we will set up the environment to be the same as gnerf. Our approach is applicable to other models as well, and camera pose optimization can be applied to general nerf models for synthetic nerf datasets.

```
# Create a conda environment named 'just_flip'
conda create --name just_flip python=3.7

# Activate the environment
conda activate just_flip

# Install requirements
pip install -r requirements.txt
```

<br/>



## Running

```
python train.py ./config/CONFIG.yaml --data_dir PATH/TO/DATASET
```

where you replace CONFIG.yaml with your config file. If you wish to monitor the training progress, you can do so with tensorboard by including the --open_tensorboard argument. It's worth noting that the default settings require approximately 13GB of GPU memory. If you are running into issues with insufficient GPU memory, it is recommended to lower the batch size and conduct your experiments with the modified configuration.

<br/>


## Evaluation

```
python eval.py --ckpt PATH/TO/CKPT.pt --gt PATH/TO/GT.json 
```

where you replace PATH/TO/CKPT.pt with your trained model checkpoint, and PATH/TO/GT.json with the json file in NeRF-Synthetic dataset. Then, just run the  [ATE toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) on the `evaluation` directory.

<br/>



## Find Optimal CameraPose
By using the function Find_Optimal_CameraPose defined in utils.py, we can modify the init_poses_embed in posemodel.py to enable flip method and pose optimization.

<br/>

## Acknowledgements
This implementation is based on guan-meng's [gnerf].

[gnerf]: https://github.com/quan-meng/gnerf