from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, concat_imgs

import pandas as pd
import numpy as np
import nibabel as nib

import glob
import os
# mask list
masks = []
with open('masklist.txt', 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        masks.append(line.strip())

# Concatenate all the effect size maps into one nifti file
language = 'CN'
# subjects = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32]
subjects = [13]
runs= [5,6,7,8,9,10,11,12,13]

for subject in subjects:
    
    for run in runs:
        
        img_path = f'1stGLM/sub-{language}{subject:0>3d}/run-{run:0>2d}'
        
        image_list = [load_img(os.path.join(img_path, f)) for f in os.listdir(img_path) if f.endswith('effect_size.nii.gz')]
            
        big_nii = concat_imgs(image_list)
        print('Done concatenating images')
        print(big_nii.dataobj.shape)



        betas = []
        for idx, mask in enumerate(masks):
            print(f"Working on {mask}")
                    
            mask_file = f'masks/{mask}.nii.gz'
            
            masker = NiftiMasker(mask_img=mask_file, standardize=False, detrend=False,
                            memory="nilearn_cache", memory_level=2)
            
            beta = masker.fit_transform(big_nii).mean(axis=1)
            print(beta.shape)
            betas.append(beta)

        betas_df = pd.DataFrame(betas).T
        betas_df.columns = [masks]
        betas_df.to_csv(f'1stGLM/sub-{language}{subject:0>3d}/run-{(run-4):0>2d}-GLM_betas.csv', index=False)