# Just Flip: Flipped Observation Generation and Optimization for Neural Radiance Fields to Cover Unobserved View

Author : Minjae Lee, Kyeongsu Kang and Hyeonwoo Yu

<br/>

## Overview
<img src="figs/overview.jpeg"  width="800" height="250">

(Left) In the conventional approach, the robot merely scans one facet of an object while in motion. This method, however, fails to render a satisfactory result from angles that the robot has yet to explore. (Right) Our unique method takes actual observations and produces their mirror images. By utilizing both the original and reflected images along with inferred camera positions, the robot learns to understand the 3D space through NeRF, even in unchartered territories. This novel technique allows us to achieve high-quality rendering results from previously unobserved perspectives, even without using images from these angles in our training set.

<br/>
