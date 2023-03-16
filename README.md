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
python train.py ./config/blender.yaml --data_dir PATH/TO/DATASET
```

If you wish to monitor the training progress, you can do so with tensorboard by including the --open_tensorboard argument. It's worth noting that the default settings require approximately 13GB of GPU memory. If you are running into issues with insufficient GPU memory, it is recommended to lower the batch size and conduct your experiments with the modified configuration.

<br/>


## Evaluation

```
python eval.py --ckpt PATH/TO/CKPT.pt --gt PATH/TO/GT.json 
```

where you replace PATH/TO/CKPT.pt with your trained model checkpoint, and PATH/TO/GT.json with the json file in NeRF-Synthetic dataset. Then, just run the [ATE toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) on the evaluation directory.

<br/>


## Result

|Scene|||PSNR||||SSIM||||LPIPS||
|----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
||Base|Ours|Ours+Pose|Base+GT|Base|Ours|Ours+Pose|Base+GT|Base|Ours|Ours+Pose|Base+GT|
|Chairs    |11.10|12.80|**14.87**|16.17|0.64|**0.74|**0.81|0.86|0.48|0.34|**0.24**|0.24|
|Ficus     |17.05|16.17|**17.73**|18.43|**0.82**|0.82|0.75|0.84|**0.21**|0.26|0.29|0.16|
|Hotdog    |13.37|13.54|**16.27**|17.20|0.67|**0.74**|0.76|0.80|0.45|0.36|**0.35**|0.30|
|Lego truck|11.25|13.53|**14.17**|23.13|0.68|**0.74**|0.77|0.88|0.48|0.39|**0.30**|0.11|
|Materials |15.24|15.87|**16.88**|17.42|0.66|**0.68**|0.80|0.81|0.42|0.42|**0.29**|0.30|
|Ship      |10.67|10.93|**17.98**|23.59|0.63|**0.64**|0.72|0.85|0.54|0.51|**0.31**|0.16|
|Mean      |13.11|13.81|**16.32**|19.32|0.68|**0.73**|0.77|0.84|0.43|0.38|**0.30**|0.21|

Quantitative comparisons of our flip methods with baseline method and baseline with ground-truth method on the NeRF synthetic dataset. Ground-truth image is an image observed from a viewpoint that the robot has not explored. Our method shows improved performance, even without estimating the flipped camera poses. In addition, we demonstrate that applying camera pose estimation to flipped images yields a more substantial improvement in performance compared to using the simple flip method.

<br>


## Acknowledgements
This implementation is based on guan-meng's [gnerf].

[gnerf]: https://github.com/quan-meng/gnerf