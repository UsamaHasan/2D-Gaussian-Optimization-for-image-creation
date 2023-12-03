import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from utils import *
import argparse
import os
class Gaussian2D(nn.Module):
    def __init__(self, img_s, num_gaussians):
        super(Gaussian2D, self).__init__()
        self.image_size = img_s.shape
        self.num_gaussians = num_gaussians
        self.grid_size = math.floor(math.sqrt(num_gaussians))
        # Parameters for Gaussians
        x_ = torch.linspace(0, self.image_size[0], self.grid_size) /  self.image_size[0]
        y_ = torch.linspace(0, self.image_size[1], self.grid_size) /  self.image_size[1]
        x_ , y_ = torch.meshgrid(x_, y_)
        means = torch.stack((x_, y_), dim=2).view(self.num_gaussians,2,1)
        covariances = torch.randn(num_gaussians, 2, 2)
        self.means = nn.Parameter(means)
        self.covariances = nn.Parameter(covariances)
        
        
        self.colors = nn.Parameter(torch.rand(self.num_gaussians, 4))  # gaussians x RGBA
        
    def forward(self):
        # Initialize image
        rgb = torch.sigmoid(self.colors[...,:3]) # RGB color
        alpha = self.colors[...,3]  # Alpha value
        # Apply Gaussian to RGB channels
        gaussian_vals = gaussian_2d_batch(self.means,self.covariances,self.image_size)
        gaussian_rgb =  rgb.view(*rgb.shape[:1],1,1,rgb.shape[1]) \
              * gaussian_vals.unsqueeze(-1).repeat(1,1,1,1,3).squeeze(0)
        # Alpha blending
        image = alpha.repeat(3,1,1,1).permute(3,2,1,0) * gaussian_rgb 
        image = image.sum(0)
        # Normalize image
        #image = torch.clamp(image, 0, 1)
        return image

if __name__ == '__main__':
    args= argparse.ArgumentParser()
    args.add_argument("--num_gaussians", type=int, default=100)
    args.add_argument("--image_path", type=str, default="canvas.png")
    args.add_argument("--no_iterations", type=int, default=1000)

    args = args.parse_args()
    results_path = "results"
    try:
        os.makedirs(results_path, exist_ok=True)
    except:
        raise Exception("Cannot create results folder")
    
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img).float().to("cuda")
    num_gaussians = args.num_gaussians

    model = Gaussian2D(img_tensor, num_gaussians).to("cuda")
    loss_l1 = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.autograd.set_detect_anomaly(True):
        for i in range(1000):
            # Forward pass
            reconstructed_image = model()
            loss_value = 2 * loss_l2(img_tensor, reconstructed_image) + 2 * loss_l1(img_tensor, reconstructed_image) 
            #+ 0.01 * torch.linalg.norm(model.covariances, dim=1, ord=2).sum()
            # Backward pass
            opt.zero_grad()  
            loss_value.backward()
            opt.step()            
            if(i%10 == 0):
                cv2.imwrite(os.path.join(results_path,f"{str(i)}.png"), \
                        cv2.cvtColor(reconstructed_image.detach().cpu().numpy()*255,cv2.COLOR_RGB2BGR))
            if (loss_value.item() < 0.05):
                break
            # print loss
            print("Epoch: {}, Loss: {:.5f}".format(i, loss_value.item()))

    cv2.imwrite(f"reconstructed_image.png in {i} iterations", reconstructed_image.detach().cpu().numpy()*255)

