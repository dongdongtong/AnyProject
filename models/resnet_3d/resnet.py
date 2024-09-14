import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence

from ..build import MODEL_REGISTRY


def get_inplanes(inplane_ratio=1):
    return [64 * inplane_ratio, 128 * inplane_ratio, 256 * inplane_ratio, 512 * inplane_ratio]


def conv3x3x3(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv1x1x1(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.in1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.in2 = nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.in1 = nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.in2 = nn.InstanceNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.in3 = nn.InstanceNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 in_channels=3,
                 num_classes: int = 2,
                 strides: Sequence[list] | Sequence[int] = (2, 2, 2, 2, 2),
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 **kwargs):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(in_channels,
                               self.in_planes,
                               kernel_size=7,
                               stride=strides[0],
                               padding=3,
                               bias=True)
        self.in1 = nn.InstanceNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=strides[1], padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=strides[2])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=strides[3])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=strides[4])
        self.num_features = block_inplanes[3] * block.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1)):
        downsample = None
        if (torch.tensor(stride) == 2).sum() > 0 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.InstanceNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
            
        return x

    def forward(self, x, need_fp=False):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        if need_fp:
            out = torch.cat([out, nn.Dropout(0.5)(out), nn.Dropout(0.5)(out)])
            out = self.fc(out)
            
            return out, x
            
        out = self.fc(out)

        # return out, x
        return out


@MODEL_REGISTRY.register()
def resnet3d(cfg, **kwargs):
    model_arch = cfg.MODEL.BACKBONE.NAME
    in_channels = cfg.MODEL.BACKBONE.IN_CHANNELS
    out_channels = cfg.MODEL.HEAD.OUT_CHANNELS
    strides = cfg.MODEL.BACKBONE.STRIDES
    inplane_ratio = cfg.MODEL.BACKBONE.INPLANE_RATIO
    layers = cfg.MODEL.BACKBONE.LAYERS
    
    block = BasicBlock if "10" in model_arch or "18" in model_arch or "34" in model_arch else Bottleneck
    
    model = ResNet(
        block, 
        layers, 
        get_inplanes(inplane_ratio), 
        in_channels=in_channels, 
        num_classes=out_channels, 
        strides=strides,
    )
    
    # inplane_ratio = kwargs.get("inplane_ratio", 1.0)
    # if model_arch == "res10":
    #     model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res18":
    #     model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res34":
    #     model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res50":
    #     model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res101":
    #     model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res152":
    #     model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # elif model_arch == "res200":
    #     model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(inplane_ratio), in_channels=in_channels, num_classes=num_class, layout=layout)
    # else:
    #     raise Exception(f"Resnet model of {model_arch} not implemented!!. You can choose one of [10, 18, 34, 50, 101, 152, 200]")

    return model