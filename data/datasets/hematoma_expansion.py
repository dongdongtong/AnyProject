"""Created by Dingsd on 09/13/2024 22:21
"""

from collections.abc import Sequence
import os
from os.path import basename, dirname, join

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import cv2
import random
from monai.data.dataset import CacheDataset
from monai.networks import one_hot
from monai.data import MetaTensor
from sklearn.model_selection import StratifiedKFold

from .base_dataset import DatasetBase, PairDatum
from .build import DATASET_REGISTRY, DATASET_WRAPPER_REGISTRY
    

import os
import os.path as osp
import random
from copy import deepcopy

from typing import Dict, List

from utils import read_json, mkdir_if_missing, write_json


@DATASET_REGISTRY.register()
class HematomaExpansionDataset(DatasetBase):
    # generate data list and saved!
    # make a data structure for hematoma segmentation
    # train val test split
    
    spacing = (0.48828101, 0.48828101, 4.5)   # this spacing is the resampled spacing.
    
    _lab2cname = {
        0: 'non expansion',
        1: 'expansion',
    }
    
    _classnames = ['non expansion', 'expansion',]
    
    _num_classes = 2
    
    dataset_json_file = "hematoma_fingerprint_grouped.json"
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.dataset_dir = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        hematoma_data_json_path = os.path.join(self.dataset_dir, self.dataset_json_file)
        hematoma_data_json = read_json(hematoma_data_json_path)
        
        train_json = hematoma_data_json['training'] + hematoma_data_json['validation']
        labels = [item['expansion_label'] for item in train_json]
        test_json = hematoma_data_json['testing']
        
        # we do cross-validation on training data
        seed = cfg.SEED
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        for fold, (train_index, val_index) in enumerate(skfold.split(train_json, labels)):
            train_data = [train_json[i] for i in train_index]
            val_data = [train_json[i] for i in val_index]
            
            # save the data to json
            fold_json_path = os.path.join(self.dataset_dir, f'he_fold_{fold}.json')
            
            fold_json = {
                "fold": fold,
                "numTraining": len(train_data),
                "numValidation": len(val_data),
                "numTesting": len(test_json),
                "training": train_data,
                "validation": val_data,
                "testing": test_json
            }
            
            if os.path.exists(fold_json_path):
                print(f'Hematoma Expansion Stratified Fold {fold} already exists!')
            else:
                write_json(fold_json, fold_json_path)
        
        self.source_domains = cfg.DATASET.SOURCE_DOMAINS
        domain = self.source_domains[0]
        
        # load the data from json
        fold = cfg.DATASET.FOLD
        print(f"Loading dataset (hematoma expansion) stratified fold {fold}...")
        fold_json_path = os.path.join(self.dataset_dir, f'fold_{fold}.json')
        fold_json = read_json(fold_json_path)
        print(f"Fold {fold} loaded! Training data: {fold_json['numTraining']}, Validation data: {fold_json['numValidation']}, Testing data: {fold_json['numTesting']}")
        
        train_x = self.read_data(fold_json['training'], domain)
        
        valid_x = self.read_data(fold_json['validation'], domain)
        
        test_x = self.read_data(fold_json['testing'], domain)
        
        super().__init__(train_x=train_x, train_u=None, val=valid_x, test=test_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data(self, data_json: List[Dict], domain: str) -> List[PairDatum]:
        data = []
        
        for item in data_json:
            baseline_image_path = osp.join(self.dataset_dir, item['baseline_image'])
            baseline_label_path = osp.join(self.dataset_dir, item['baseline_label'])
            followup_image_path = osp.join(self.dataset_dir, item['24h_image'])
            followup_label_path = osp.join(self.dataset_dir, item['24h_label'])
            
            hematoma_expansion_label = item['expansion_label']
            classname = self._classnames[hematoma_expansion_label]
            
            data.append(
                PairDatum(
                    baseline_image_path, 
                    followup_image_path,
                    baseline_label_path, 
                    followup_label_path,
                    hematoma_expansion_label, 
                    domain, 
                    classname
                )
            )
        
        return data
        

@DATASET_WRAPPER_REGISTRY.register()
class HematomaExpansionWrapper(Dataset):

    def __init__(self, cfg, data_source, base_transform=None, other_transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.base_transform = base_transform
        self.other_transform = other_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        output_dict = {
            "img": item.impath,
            "seg": item.segpath,
            "domain": item.domain,
            "label": item.label,
            "path": item.impath,
            "index": idx
        }
        
        output_dict = self.base_transform(output_dict)

        if self.other_transform is not None:
            if isinstance(self.other_transform, (list, tuple)):
                for i, tfm in enumerate(self.other_transform):
                    img, seg = self._transform(tfm, deepcopy(output_dict))
                    
                    img_keyname = "img"
                    seg_keyname = "seg"
                    if (i + 1) > 1:
                        img_keyname += str(i + 1)
                        seg_keyname += str(i + 1)
                    
                    output_dict[img_keyname] = img
                    output_dict[seg_keyname] = seg
            else:
                img, seg = self._transform(self.other_transform, deepcopy(output_dict))
                output_dict["img"] = img
                output_dict["seg"] = seg
        else:
            return output_dict

        return output_dict

    def _transform(self, tfm, input_dict):
        tfed_dict = tfm(input_dict)
        
        if isinstance(tfed_dict, dict):
            tfed_img = tfed_dict['img']
            tfed_seg = tfed_dict['seg']
        elif isinstance(tfed_dict, list):  # multi patch training, do not use list to avoid memory leak
            len_tfed_dict = len(tfed_dict)
            img_size = tfed_dict[0]['img'].shape
            tfed_img = torch.zeros(len_tfed_dict, *img_size)
            tfed_seg = torch.zeros(len_tfed_dict, *img_size)

            for i, tfed_patch_dict in enumerate(tfed_dict):
                tfed_img[i] = tfed_patch_dict['img']
                tfed_seg[i] = tfed_patch_dict['seg']
        else:
            raise ValueError(f"Unknown type of tfed_dict: {type(tfed_dict)}")
        
        return tfed_img, tfed_seg


@DATASET_WRAPPER_REGISTRY.register()
class HematomaExpansionCachedWrapper(CacheDataset):
    def __init__(self, cfg, data_source, base_transform=None, other_transform=None, is_train=False):
        cache_num = len(data_source) if cfg.DATASET_WRAPPER.CACHE_NUM == -1 else cfg.DATASET_WRAPPER.CACHE_NUM
        num_workers = os.cpu_count()
        
        data_json = [
            {
                "img": item.impath,
                "seg": item.segpath,
                "domain": item.domain,
                "label": item.label,
                "path": item.impath,
                "index": idx
            }
            for idx, item in enumerate(data_source)
        ]
        
        self.data_source = data_source
        
        self.base_transform = base_transform
        self.other_transform = other_transform
        self.is_train = is_train
        
        super().__init__(data_json, base_transform, cache_num=cache_num, num_workers=num_workers)

    def __getitem__(self, idx):
        output_dict = self._cache[idx]

        if self.other_transform is not None:
            if isinstance(self.other_transform, (list, tuple)):
                for i, tfm in enumerate(self.other_transform):
                    img, seg = self._transform(tfm, deepcopy(output_dict))
                    
                    img_keyname = "img"
                    seg_keyname = "seg"
                    if (i + 1) > 1:
                        img_keyname += str(i + 1)
                        seg_keyname += str(i + 1)
                    
                    output_dict[img_keyname] = img
                    output_dict[seg_keyname] = seg
            else:
                img, seg = self._transform(self.other_transform, deepcopy(output_dict))
                output_dict["img"] = img
                output_dict["seg"] = seg
        else:
            return output_dict

        return output_dict

    def _transform(self, tfm, input_dict):
        tfed_dict = tfm(input_dict)
        
        if isinstance(tfed_dict, dict):
            tfed_img = tfed_dict['img']
            tfed_seg = tfed_dict['seg']
        elif isinstance(tfed_dict, list):  # multi patch training
            len_tfed_dict = len(tfed_dict)
            img_size = tfed_dict[0]['img'].shape
            tfed_img = torch.zeros(len_tfed_dict, *img_size)
            tfed_seg = torch.zeros(len_tfed_dict, *img_size)

            for i, tfed_patch_dict in enumerate(tfed_dict):
                tfed_img[i] = tfed_patch_dict['img']
                tfed_seg[i] = tfed_patch_dict['seg']
        else:
            raise ValueError(f"Unknown type of tfed_dict: {type(tfed_dict)}")
        
        return tfed_img, tfed_seg