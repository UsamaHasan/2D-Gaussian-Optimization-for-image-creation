# 2D Gaussian Optimization for Image Creation

## Introduction
The following repo contains the code of take home assessment task. The task is to create an image using 2D Gaussian with mean vector (Nx2) and a covariance matrix (Nx2x2) where N is the number of Gaussian to fit the image. 

## Requirements
The only requirement is pytorch and opencv, which was used to load and save the reconstructed image.
Further I have tested the code on Pytorch 1.3.1 on ubuntu 20.04.

## Testing
``
python 2d_gaussian.py --num_gaussians $NUM_OF_GAUSSIAN_TO_FIT --image_path $PATH_TO_INPUT_IMAGE
``

## Implementation 
I have tried to write multiple implementations due to the fact the covariance matrix that we need to optimize here is a (2x2), and we need to make sure that it's positive semi-definite, so for that, I have used multiple hacks to solve that like add small nudge in the diagonal based on the smallest eigenvalue. Also as authors did in the original paper, I divided the covariance matrix into a rotation matrix and scale vector for each Gaussian but that also doesn't solve the (2x2) scenario as mentioned by the author as well in the paper. Finally, I have tried to add comments to make it understandable as much as I could.

## Results.
Input Image: ![]([https://github.com/Your_Repository_Name/Your_GIF_Name.gif](https://github.com/UsamaHasan/2D-Gaussian-Optimization-for-image-creation/blob/master/canvas.png) 
Output Generation: ![](https://github.com/UsamaHasan/2D-Gaussian-Optimization-for-image-creation/blob/master/res/optim_output.gif)
