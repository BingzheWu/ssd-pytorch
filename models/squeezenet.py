
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.models.squeezenet import SqueezeNet, Fire
from torchvision.models.resnet import ResNet, BasicBlock

import sys
sys.path.append("../")
from layers import *
from data import v_resnet,v3,v_sq
import os

class SSD(nn.Module):
    def __init__(self, phase, base, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.base = base
        self.priorbox = PriorBox(v_resnet)
        self.priors = Variable(self.priorbox.forward(), volatile = True)
        self.size = 300
        self.make_loc_and_conf()
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 3, 10, 0.01, 0.45)
    def make_loc_and_conf(self, out_channels = [128, 256, 512, 512, 512, 512], num_boxes = [3,6,6,6,6, 6]):
        loc_layers = []
        conf_layers = []
        for out_channel, num in zip(out_channels, num_boxes):
            loc_layers.append(nn.Conv2d(out_channel, num*4, kernel_size = 3, padding = 1))
            conf_layers.append(nn.Conv2d(out_channel, num*self.num_classes, kernel_size = 3, padding = 1))
        self.loc= nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
    def forward(self, x):
        features = self.base(x)
        loc = list()
        conf = list()
        for (feature, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(feature).permute(0,2,3,1).contiguous())
            conf.append(c(feature).permute(0,2,3,1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
def conv_bn(inp, oup, stride, k_size = 3, paddings = 1):
    return nn.Sequential(
            nn.Conv2d(inp, oup, k_size, stride, paddings),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace = True)
        )
class sq_feature(nn.Module):
    def __init__(self, verion = 1.0, num_classes = 1000):
        super(sq_feature, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride = 2, ceil_mode = True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
        )
        #self.Fire1 = Fire(256, 32, 128, 128)
        self.Fire1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.Fire2 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire(512, 96, 384, 384),
        )
        self.Fire3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            Fire(768, 96, 384, 384),
        )
        self.Fire4 = nn.Sequential(
            conv_bn(768, 128, 1, 1),
            conv_bn(128, 128, 2),
        )
        self.Fire5 = nn.Sequential(
            conv_bn(128, 64, 1, 1),
            conv_bn(64, 128, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        source_features = []
        x = self.conv1(x)
        source_features.append(x)
        x = self.Fire1(x)
        source_features.append(x)
        x = self.Fire2(x)
        source_features.append(x)
        x = self.Fire3(x)
        source_features.append(x)
        x = self.Fire4(x)
        source_features.append(x)
        x = self.Fire5(x)
        source_features.append(x)
        return source_features




def test_feature_extrator():
    sq = sq_feature()
    #resnet = resnet_feature(BasicBlock, [2,2,2,2])
    
    inputs = torch.zeros((1,3, 300,300))
    inputs = Variable(inputs)
    features = sq(inputs)
    for feature in features:
        print(feature.size())

def test_SSD():
    resnet = resnet_feature(BasicBlock, [3,4,6,3])
    ssd_detector = SSD('train', resnet, 3)
    inputs = torch.zeros((1,3, 300,300))
    inputs = Variable(inputs)
    o = resnet.forward(inputs)
    for x in o:
        print(x.size())
if __name__ == '__main__':
    test_feature_extrator()
    



