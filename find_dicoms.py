import pandas as pd
import os
from os.path import join, exists, dirname
import nibabel as nib
import numpy as np
from shutil import copytree, copyfile, copy, copy2


dicom2nii_df = pd.read_csv('/data/dingshaodong/datasets/help_dcm2nii_manufacturer_info.csv', converters={'patient_id': str})


nii_img_root_dir = "/data/dingshaodong/datasets/hematoma_nnunet/baseline_images"
nii_label_root_dir = "/data/dingshaodong/datasets/hematoma_nnunet/baseline_labels"

selected_count = 20
count = 0

output_dir = "/data/dingshaodong/datasets/keep_random_task_data/intraventricular_data"
for row in dicom2nii_df.itertuples():
    pid = row.patient_id
    match_type = int(row.dcm_nii_match)
    
    if match_type == 1:
        img_path = join(nii_img_root_dir, pid + "_0000.nii.gz")
        seg_path = join(nii_label_root_dir, pid + ".nii.gz")
        
        if exists(seg_path) and exists(img_path):
            seg_nii = nib.load(seg_path).get_fdata()
            
            if len(np.unique(seg_nii)) == 3:
                dcm_dir = row.matched_dcms_dir_path
                
                out_dcm_dir = join(output_dir, "baseline_images_dcm", pid)
                os.makedirs(out_dcm_dir, exist_ok=True)
                copytree(dcm_dir, out_dcm_dir, dirs_exist_ok=True)
                
                out_image_path = join(output_dir, "baseline_images_nii", pid + "_0000.nii.gz")
                out_label_path = join(output_dir, "baseline_labels_nii", pid + ".nii.gz")
                os.makedirs(dirname(out_label_path), exist_ok=True)
                os.makedirs(dirname(out_image_path), exist_ok=True)
                copyfile(seg_path, out_label_path)
                copyfile(img_path, out_image_path)
                
                count += 1
                
                if count == selected_count:
                    break
                
                