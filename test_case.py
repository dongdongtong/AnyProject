from data.transforms import hematoma_seg_transforms
from yacs.config import CfgNode as CN
from torchvision.utils import make_grid, save_image

cfg = CN()
cfg.INPUT = CN()
cfg.INPUT.SIZE = (256, 256, 24)
cfg.INPUT.ROI_SIZE = (96, 96, 12)
cfg.INPUT.INTENSITY_RANGE = (0, 100)

base_transforms = hematoma_seg_transforms(cfg, is_train=False, only_other_transforms=False)
extra_transforms = hematoma_seg_transforms(cfg, is_train=True, only_other_transforms=True)

example_nii_path = "/dingshaodong/projects/hematoma_meta/data/preprocessed/resampled/baseline_images/1228060_0000.nii.gz"
example_seg_path = "/dingshaodong/projects/hematoma_meta/data/preprocessed/resampled/baseline_labels/1228060.nii.gz"

input_dict = {
    'img': example_nii_path,
    'seg': example_seg_path
}

transformed_dict = base_transforms(input_dict)
# print(transformed_dict)

init_img = transformed_dict['img']  # shape (1, W, H, D)
init_seg = transformed_dict['seg']  # shape (1, W, H, D)
init_img_grid = make_grid(init_img.permute(3, 0, 1, 2), nrow=4, padding=0)
init_seg_grid = make_grid(init_seg.permute(3, 0, 1, 2), nrow=4, padding=0)
save_image(init_img_grid, 'init_img_grid.png')
save_image(init_seg_grid, 'init_seg_grid.png')

transformed_dict = extra_transforms(transformed_dict)
# print(transformed_dict)  # a list of (1, 96, 96, 12)

for i, roi_dict in enumerate(transformed_dict):
    img_grid = make_grid(roi_dict['img'].permute(3, 0, 1, 2), nrow=4, padding=0)
    seg_grid = make_grid(roi_dict['seg'].permute(3, 0, 1, 2), nrow=4, padding=0)
    save_image(img_grid, f'transformed_roi_img_grid_{i}.png')
    save_image(seg_grid, f'transformed_roi_seg_grid_{i}.png')


# print os RAM memory
import os

print(f"Total RAM memory size: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 / 1024 / 1024} G")  # 754 G