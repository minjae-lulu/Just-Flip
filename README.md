# Just Flip: Flipped Observation Generation and Optimization for Neural Radiance Fields to Cover Unobserved View

Implementation of our method. Our data agumentation approach is flippimg observed images, and estimating flipped camera 6DoF poses.
<br/>

## Overview
<img src="figs/overview.jpeg"  width="700" height="200">
(Left) the baseline approach where the robot only observes one side of an object while driving. This case does not yield good rendering results in unobserved views that the robot has not explored. (Right) our method generates the flipped observations from the actual observations. The robot exploits both input images and flipped images and estimated camera poses to learn 3D space using NeRF for unexplored regions as well. 

<br/><br/>

## Enviroment setup
Our baseline model is gnerf. So follow the instructions for setting up the environment in the gnerf folder. Our method could be applicable to other models, such as [barf].

[barf]: https://github.com/chenhsuanlin/bundle-adjusting-NeRF


<br/><br/>

## Flipping image
<!-- test
 ```
aaa
 ``` -->

<br/><br/>

## Pose estimation


<br/><br/>


## BibTeX

<br/><br/>

## Acknowledgements
This implementation is based on guan-meng's [gnerf].

[gnerf]: https://github.com/quan-meng/gnerf