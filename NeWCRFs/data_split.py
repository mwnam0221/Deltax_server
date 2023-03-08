import glob 
import cv2
from tqdm import tqdm
import os 
import natsort
from sklearn.model_selection import train_test_split

img_list=natsort.natsorted(glob.glob('/hdd/hoa/nyu_data/data/nyu2_train/*/*.jpg'))
depth_list=natsort.natsorted(glob.glob('/hdd/hoa/nyu_data/data/nyu2_train/*/*.png'))

x_train_img, x_valid_img = train_test_split(img_list, test_size = 0.2, random_state=42)
x_train_depth, x_valid_depth = train_test_split(depth_list, test_size = 0.2, random_state=42)

print(x_train_depth)

with open('/home/dan/NeWCRFs/temp_train.txt','w') as f:
    for line in zip(x_train_img, x_train_depth):
        f.write(line[0]+' '+line[1])
        f.write('\n')


with open('/home/dan/NeWCRFs/temp_valid.txt','w') as f:
    for line in zip(x_valid_img, x_valid_depth):
        f.write(line[0]+' '+line[1])
        f.write('\n')


