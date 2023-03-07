
import shutil

with open('./val_files_CVPR.txt', 'r') as f:
    samples = f.readlines()


for i, sample in enumerate(samples):
    sample = sample[:-1]
    print(sample)
    print('/hdd/team_2/syns_patches/'+sample, f'/home/dan/Deltax_server/MIM-Depth-Estimation/data/{i}.png')
    
    shutil.copy('/hdd/team_2/syns_patches/'+sample, f'/home/dan/Deltax_server/MIM-Depth-Estimation/data/{i}.png')