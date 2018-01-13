from models import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import sys
sys.path.append("../")
from layers import *
from data import v2
import os

model_urls = {
    'alexnet':'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

def build_ssd(phase, size = 300, num_classes = 21, net = 'resnet', args = None):
    if net == 'resnet':
        resnet = resnet_feature(BasicBlock, [2,2,2,2])
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = resnet.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        resnet.load_state_dict(model_dict)
        ssd_detector = SSD(phase, resnet, num_classes)
    if net == 'vgg16':
        from .ssd import build_ssd_
        ssd_detector = build_ssd_(phase, size, num_classes=num_classes)
        if phase == 'train':
            vgg_weights = torch.load(args.save_folder+args.basenet)
            ssd_detector.vgg.load_state_dict(vgg_weights)
    return ssd_detector

def test_build_ssd():
    build_ssd('train')
