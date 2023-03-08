import numpy as np
import os
import config

####Config####
ICRA= config.icra
WEIGHT = config.weight

print('config...')
print(ICRA)
print(WEIGHT)

####Dir####
save_img_dir = '/home/dan/NewCRFs/newcrfs/'
output_path = os.path.join(save_img_dir, "ensemble.npz")
####Load####
print('Load...')
dis_monodepth = np.load('/home/dan/monodepth2/results/submission_02_14_2023_17_34/dis_monodepth2.npy')
dis_pixel=np.load('/home/dan/PixelFormer/pixelformer/submission_02_14_2023_17_41/dis_pixel.npy')

####Ensemble####
print('Ensemble...')
print(len(dis_pixel))
dis_ensemble = dis_pixel *WEIGHT + dis_monodepth *(1-WEIGHT)
dis_ensemble = list(dis_ensemble)
####Save Numpy####
print('Saving...')
if ICRA:
    np.savez_compressed('/home/dan/NeWCRFs/newcrfs/ensemble.npz', data=dis_ensemble)
else:
    np.savez_compressed('/home/dan/NeWCRFs/newcrfs/ensemble.npz', pred=dis_ensemble)
