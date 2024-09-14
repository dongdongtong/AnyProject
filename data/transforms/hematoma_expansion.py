import torch

import numpy as np
import monai
import monai.transforms as tfs

from .build import TRANSFORM_REGISTRY


class TrancateIntensityd(tfs.MapTransform):
    """
    Truncate intensity values to [minv, maxv] within the image.
    """
    def __init__(self, keys, minv=0, maxv=100, image_key="image", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.keys = keys
        self.minv = minv
        self.maxv = maxv
        self.image_key = image_key

    def __call__(self, data):
        for key in self.keys:
            if key == self.image_key:
                if monai.__version__ >= "1.0.0":
                    data[key] = torch.clamp(data[key], min=self.minv, max=self.maxv)
                else:
                    data[key] = np.clip(data[key], a_min=self.minv, a_max=self.maxv)

        return data


@TRANSFORM_REGISTRY.register()
def hematoma_expansion_transforms(cfg, is_train=True, only_other_transforms=False):
    image_size = cfg.INPUT.SIZE
    resize_size = cfg.INPUT.RESIZE_SIZE  # the resize size of the entire brain
    intensity_range = cfg.INPUT.INTENSITY_RANGE  # clip image intensity to [minv, maxv]
    
    img_key = 'img'
    seg_key = 'seg'
    keys = [img_key, seg_key]
    
    base_transforms = [
        tfs.LoadImaged(keys=keys),
        tfs.EnsureChannelFirstd(keys=keys),
        tfs.Orientationd(keys=keys, axcodes='RAS'),
        tfs.ResizeWithPadOrCropd(keys=keys, spatial_size=image_size, mode='edge', method='symmetric'),
        TrancateIntensityd(keys=keys, minv=intensity_range[0], maxv=intensity_range[1], image_key=img_key),
        tfs.ScaleIntensityd(keys=img_key, minv=0.0, maxv=1.0),
        tfs.Resized(keys=keys, spatial_size=resize_size, mode=['area', 'nearest']),
    ]
    
    if not is_train:
        return tfs.Compose(base_transforms)
    
    spatial_transforms = [
        tfs.RandFlipd(keys=keys, prob=0.5, spatial_axis=[0, 1, 2]),
        tfs.RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1), ),
        tfs.RandSimulateLowResolutiond(keys=keys, prob=0.5, zoom_range=(0.5, 1.0)),
        tfs.RandRotated(keys=keys, prob=0.5, range_z=30, mode='bilinear', align_corners=False),  # only rotate in-plane matrix, range_x is incorrect
    ]
    
    intensity_transforms = [
        tfs.RandScaleIntensityd(keys=img_key, factors=0.1, prob=0.5),
        tfs.RandAdjustContrastd(keys=img_key, prob=0.5),
        tfs.RandShiftIntensityd(keys=img_key, offsets=0.1, prob=0.5),
        tfs.RandGaussianNoised(keys=img_key, prob=0.5),
        tfs.RandGaussianSmoothd(keys=img_key, prob=0.5),
    ]
    
    if only_other_transforms:
        transforms = spatial_transforms + intensity_transforms
    else:
        transforms = base_transforms + spatial_transforms + intensity_transforms
    
    return tfs.Compose(transforms)


@TRANSFORM_REGISTRY.register()
def hematoma_expansion_base_transforms(cfg, is_train=True, only_other_transforms=False):
    image_size = cfg.INPUT.SIZE
    resize_size = cfg.INPUT.RESIZE_SIZE  # the resize size of the entire brain
    intensity_range = cfg.INPUT.INTENSITY_RANGE  # clip image intensity to [minv, maxv]
    
    img_key = 'img'
    seg_key = 'seg'
    keys = [img_key, seg_key]
    
    base_transforms = [
        tfs.LoadImaged(keys=keys),
        tfs.EnsureChannelFirstd(keys=keys),
        tfs.Orientationd(keys=keys, axcodes='RAS'),
        tfs.ResizeWithPadOrCropd(keys=keys, spatial_size=image_size, mode='edge', method='symmetric'),
        TrancateIntensityd(keys=keys, minv=intensity_range[0], maxv=intensity_range[1], image_key=img_key),
        tfs.ScaleIntensityd(keys=img_key, minv=0.0, maxv=1.0),
        tfs.Resized(keys=keys, spatial_size=resize_size, mode=['area', 'nearest']),
    ]
    
    if not is_train:
        return tfs.Compose(base_transforms)
    
    return None