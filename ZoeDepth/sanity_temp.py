import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

# Import necessary modules
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint
import glob
from tqdm import tqdm 
import natsort
import cv2
import os
from datetime import datetime
import config

MODEL_TYPE=config.model_type

# Get the current timestamp to create a unique directory for saving the results
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
save_img_dir = f'./results/submission_{timestamp}'
if not os.path.isdir(save_img_dir):
    os.makedirs(save_img_dir)


def depth_value_to_depth_image(depth_values, disp=True):
    depth_values = cv2.normalize(depth_values, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    if disp:
        depth = 1/depth_values
        depth = (depth).astype(np.uint8)
    else:
        depth = (depth_values * 255).astype(np.uint8)
        
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    
    return depth

# Function to compute the final disparity map from the predicted depth map
def final_disp(depth_values, is_gt = False):
    # Normalize the depth values
    depth_values = cv2.normalize(depth_values, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # Compute the disparity map from the depth map
    disp = 1/depth_values
    return disp

# Check if CUDA is available and set the device accordingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

# Load the configuration for zoedepth model inference
if MODEL_TYPE:
    conf = get_config("zoedepth", "infer")
else:
    conf = get_config("zoedepth", "infer", config_version="kitti")
print("Config:")
pprint(conf)

# Build the zoedepth model and move it to the device
model = build_model(conf).to(DEVICE)
model.eval()

# Load images from the specified directory and run inference on them
print("-"*20 + " Testing on an indoor scene from url " + "-"*20)
# Test img
pred_depths = []
img_lists=natsort.natsorted(glob.glob('/home/dan/Deltax_server/MIM-Depth-Estimation/data/*.png'))
for i,img_path in enumerate(img_lists):
    # Read the image using Pillow library and get its original size
    img = Image.open(img_path)
    orig_size = img.size
    # Convert the image to tensor and move it to the device
    X = ToTensor()(img)
    X = X.unsqueeze(0).to(DEVICE)

    print("X.shape", X.shape)
    print("predicting")

    # Run inference on the input tensor
    with torch.no_grad():
        out = model.infer(X).cpu()

        # Compute the final disparity map from the predicted depth map and append to the list
        final_disp_pred=final_disp(out.numpy().squeeze())
        pred_depths.append(final_disp_pred)
                
    print("output.shape", out.shape)
    pred = Image.fromarray(colorize(out))
    # Stack img and pred side by side for comparison and save
    pred = pred.resize(orig_size, Image.ANTIALIAS)
    stacked = Image.new("RGB", (orig_size[0]*2, orig_size[1]))
    stacked.paste(img, (0, 0))
    stacked.paste(pred, (orig_size[0], 0))

    stacked.save(f"{save_img_dir}/pred_{i}.png")
    
    cv2.imwrite(f'{save_img_dir}/pred_{i}.jpg', depth_value_to_depth_image(out.numpy().squeeze(), disp=False))

    print("saved pred.png")

#saving
print(len(pred_depths))
output_path = os.path.join(save_img_dir, "pred.npz")
np.savez_compressed(output_path, pred=pred_depths)
