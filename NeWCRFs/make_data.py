import glob 
import cv2
from tqdm import tqdm
import os 
import natsort


file = open("/home/dan/NeWCRFs/data_splits/val_files_CVPR.txt", "r")
print(file)
image_path = file.readlines()
image_pathes = ['/hdd/team_2/syns_patches/' + x.strip() for x in image_path]
print(image_pathes)