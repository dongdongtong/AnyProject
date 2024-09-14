"""Created by Dingsd on 2024/09/13.
In trainer, we should do:
1. data loading pipeline
2. model definition
3. model training
4. model evaluation
5. model inference
"""
import time
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from functools import partial

from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

from ..build import TRAINER_REGISTRY
from ..base_trainer import TrainerX
from optim import build_optimizer, build_lr_scheduler
from models import build_model as build_modeling
from evaluation import build_evaluator

from utils import (
    MetricMeter, AverageMeter, load_pretrained_weights, 
    load_checkpoint, save_checkpoint, 
)


@TRAINER_REGISTRY.register()
class AllCombine(TrainerX):
    """We perform hematoma expansion using combined data from multiple centers.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ALLCOMBINE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        model = build_modeling(cfg)
        model.to(self.device)
        self.model = model
        
        # Double check needed updated parameters
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        trainable_params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters count: {sum(trainable_params)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            
        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        if cfg.TRAINER.ALLCOMBINE.PREC == "amp" or cfg.TRAINER.ALLCOMBINE.PREC == "fp32":
            self.model.float()
        else:
            self.model.half()
            
            # for name, module in self.model.named_modules():
            #     if isinstance(module, nn.LayerNorm):
            #         module.float()
        
        self.scaler = GradScaler() if cfg.TRAINER.ALLCOMBINE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def model_inference(self, image):
        out = self.model(image)
        return out
    
    def img_ce_loss(self, images, labels):
        logits = self.model(images)
        
        loss = F.cross_entropy(logits, labels)
        
        self.model_backward_and_update(loss, grad_record=True, names=['model', ])  # update all parameters
        
        return {
            "loss": loss.detach()
        }
    
    def img_seg_ce_loss(self, images, segs, labels):
        inputs = torch.cat([images, segs], dim=1)
        logits = self.model(inputs)
        
        loss = F.cross_entropy(logits, labels)
        
        self.model_backward_and_update(loss, grad_record=True, names=['model', ])  # update all parameters
        
        return {
            "loss": loss.detach()
        }
    
    def get_loss(self, batch_x):
        images, segs, labels = self.parse_batch_train(batch_x)
        
        if self.cfg.TRAINER.ALLCOMBINE.METHOD == 1:
            loss_dict = self.img_ce_loss(images, labels)
        elif self.cfg.TRAINER.ALLCOMBINE.METHOD == 2:
            loss_dict = self.img_seg_ce_loss(images, segs, labels)
        else:
            raise ValueError(f"Unknown method: {self.cfg.TRAINER.ALLCOMBINE.METHOD}")
        
        return loss_dict

    def forward_backward(self, batch_x):
        prec = self.cfg.TRAINER.ALLCOMBINE.PREC

        if prec == "amp":
            with autocast():
                out_dict = self.get_loss(batch_x)
        else:
            # with torch.autograd.set_detect_anomaly(True):
            out_dict = self.get_loss(batch_x)
        
        loss_summary = deepcopy(out_dict)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
        
    def parse_batch_train(self, batch_x):
        input_x = batch_x["img"]
        seg_x = batch_x["seg"]
        label_x = batch_x["label"]
        
        input_x = input_x.to(self.device)
        seg_x = seg_x.to(self.device)
        label_x = label_x.to(self.device)
        
        return input_x, seg_x, label_x

    def parse_batch_test(self, batch):
        input_x = batch["img"]
        seg_x = batch["seg"]
        label_x = batch["label"]
        
        input_x = input_x.to(self.device)
        seg_x = seg_x.to(self.device)
        label_x = label_x.to(self.device)
        
        if self.cfg.TRAINER.ALLCOMBINE.USE_SEG_MASK:
            inputs = torch.cat([input_x, seg_x], dim=1)
        else:
            inputs = input_x
        
        return inputs, label_x

    def model_backward_and_update(self, loss, names=None, grad_record=False, grad_clip=False, grad_tag="g"):
        names = self.get_model_names(names)
        self.model_zero_grad(names)
        
        accumulate_steps = self.cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS

        if self.cfg.TRAINER.ALLCOMBINE.PREC == "amp":
            self.scaler.scale(loss / accumulate_steps).backward()
        else:
            (loss / accumulate_steps).backward()
        
        # gradient clipping
        if grad_clip:
            for name in names:
                torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=11.0)
            
        def name_grad_available(name, update_model_names):
            available = False
            for model_name in update_model_names:
                if model_name in name:
                    available = True
                    break
            return available
        
        if self.batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0 and grad_record:
            for name, param in self.model.named_parameters():
                try:
                    if param.requires_grad and name_grad_available(name, names):
                        self._writer.add_scalar(f"grad_{grad_tag}/{name}", param.grad.norm(), global_step=self.epoch * self.num_batches)
                except Exception as e:
                    print(f"Error in recording gradient: {e}, name: {name}")
        
        if (self.batch_idx + 1) % accumulate_steps == 0 or self.batch_idx == self.num_batches - 1:
            if self.cfg.TRAINER.ALLCOMBINE.PREC == "amp":
                for name in names:
                    if self._optims[name] is not None:
                        self.scaler.step(self._optims[name])
                self.scaler.update()
            else:
                for name in names:
                    if self._optims[name] is not None:
                        self._optims[name].step()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

