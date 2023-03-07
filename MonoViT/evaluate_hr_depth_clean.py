from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import glob
import natsort
from tqdm import tqdm 
from datetime import datetime
from torchvision import transforms, datasets
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate():
    """Evaluates a pretrained model using a specified test set"""
    
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    decoder_path = os.path.join('/home/dan/Deltax_server/MonoViT/tmp/mono_model/models/hr_weights', "depth.pth")
    depth_dict = torch.load(decoder_path)
    feed_height, feed_width = depth_dict['height'], depth_dict['width']
    new_dict = {k[7:]:v for k,v in depth_dict.items()} # remove 'decoder.' prefix from keys
    depth_decoder = networks.DeepNet('mpvitnet')      
    depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()})
    depth_decoder.to(device)
    depth_decoder.eval()
    
    # result test set
    ######
    save_img_dir = f'./results/submission_{timestamp}'
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)
    output_directory = './results'
    ######
    file_name = '/home/dan/NeWCRFs/data_splits/test_files_CVPR.txt'
    with open(file_name, "r") as f:
        lines = f.readlines()
    test_image_paths = ['/hdd/team_2/syns_patches/'+x.strip() for x in natsort.natsorted(lines)]
    image_names = ['_'.join(x.split('/')[4:]) for x in test_image_paths]
    ######
    # Predict depths for test set
    pred_depths = []
    with torch.no_grad():
        for image_path, image_name in tqdm(zip(test_image_paths, image_names)):
            print(image_name)
            
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            w, h = input_image.size
            # ##############################################################
            # pad_h = ((h + 31) // 32) * 32 - h
            # pad_w = ((w + 31) // 32) * 32 - w
            # top, bottom = pad_h // 2, pad_h - pad_h // 2
            # left, right = pad_w // 2, pad_w - pad_w // 2
            # # Pad the input image using np.pad()
            # padded_image = np.pad(input_image, ((top, bottom), (left, right), (0, 0)), mode='edge')
            # output_image = padded_image[top:top+feed_height, left:left+feed_width]
            # print(padded_image.shape, output_image.shape)  #(384, 1248, 3) (320, 1024, 3)
            ##############################################################
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            # Predict depth
            input_image = input_image.to(device)              
            outputs = depth_decoder(input_image)  #(1024, 320)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (h, w), mode="bilinear", align_corners=False)
            scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
        
            # Save colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(save_img_dir,'{}'.format(image_name))
            im.save(name_dest_im)

            pred_depths.append(scaled_disp.cpu().numpy().squeeze())

        print(len(pred_depths))
        np.savez_compressed(save_img_dir+'/pred.npz', pred=pred_depths)
        output_path_disp= os.path.join(save_img_dir, "dis_monodepth2")
        np.save((output_path_disp), pred_depths)
        print('-> Done!')

if __name__ == "__main__":
    evaluate()
    
    
    