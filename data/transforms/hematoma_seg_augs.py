import torch

import monai.transforms as tfs

from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
def hematoma_seg_transforms(cfg, is_train=True, only_other_transforms=False):
    patch_size = cfg['patch_size']
    
    keys = ['image', 'label']
    image_key_idx = 0
    
    base_transforms = [
        tfs.LoadImaged(keys=keys),
        tfs.EnsureChannelFirstd(keys=keys),
        tfs.Orientationd(keys=keys, axcodes='RAS'),
        tfs.Compose([  # clip image to [0, 100]
            tfs.ThresholdIntensityd(keys=keys[image_key_idx], threshold=0, cval=0, above=False),
            tfs.ThresholdIntensityd(keys=keys[image_key_idx], threshold=100, cval=0, above=True),
        ]),
        tfs.ScaleIntensityd(keys=keys[image_key_idx], minv=0.0, maxv=1.0),
    ]
    
    if not is_train:
        return tfs.Compose(base_transforms)
    
    spatial_transforms = [
        tfs.RandFlipd(keys=keys, prob=0.5, spatial_axis=[0, 1, 2]),
        tfs.RandRotate90d(keys=keys, prob=0.5, spatial_axes=[0, 1]),
        tfs.RandSimulateLowResolutiond(keys=keys, prob=0.5, scale_range=(0.5, 1.0)),
        tfs.RandRotated(keys=keys, prob=0.5, range_x=30, mode='bilinear', align_corners=False),  # only rotate in-plane matrix
        tfs.RandCropByPosNegLabeld(
            keys=keys,
            label_key='label',
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key=keys[image_key_idx],
            image_threshold=0,
            allow_smaller=True,
        )
    ]
    
    intensity_transforms = [
        tfs.RandScaleIntensityd(keys=keys[image_key_idx], factors=0.1, prob=0.5),
        tfs.RandAdjustContrastd(keys=keys[image_key_idx], prob=0.5),
        tfs.RandShiftIntensityd(keys=keys[image_key_idx], offsets=0.1, prob=0.5),
        tfs.RandGaussianNoised(keys=keys[image_key_idx], prob=0.5),
        tfs.RandGaussianSmoothd(keys=keys[image_key_idx], prob=0.5),
    ]
    
    if only_other_transforms:
        transforms = spatial_transforms + intensity_transforms
    else:
        transforms = base_transforms + spatial_transforms + intensity_transforms
    
    return tfs.Compose(transforms)
    
    