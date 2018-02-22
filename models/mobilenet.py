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
from data import v_resnet,v3,v_mobilenet
import os

class SSD(nn.Module):
    def __init__(self, phase, base, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.base = base
        self.priorbox = PriorBox(v_mobilenet)
        self.priors = Variable(self.priorbox.forward(), volatile = True)
        self.size = 300
        self.make_loc_and_conf()
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 3, 10, 0.01, 0.45)
    def make_loc_and_conf(self, out_channels = [512, 1024, 512, 256, 256, 128], num_boxes = [3,6,6,6,6, 6]):
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
class mobilenet_v1(nn.Module):
    def __init__(self):
        super(mobilenet_v1, self).__init__()
        def conv_bn(inp, oup, stride, k_size = 3, paddings = 1):
            return nn.Sequential(
            nn.Conv2d(inp, oup, k_size, stride, paddings),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace = True)
        )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups = inp, bias = False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace = True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias = False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace = True)
        )
        self.init_conv = conv_bn(3, 32, 2)
        self.layer1 = nn.Sequential(
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2)
        )
        self.layer2 = nn.Sequential(
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2)
        )
        self.layer3 = nn.Sequential(
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2)
        )
        self.layer4 = nn.Sequential(*[conv_dw(512, 512, 1) for i in range(5)])
        self.layer5 = conv_dw(512, 1024, 2)
        self.layer6 = conv_dw(1024, 1024, 1)
        self.layer7 = nn.Sequential(
            conv_bn(1024, 256, 1, 1, 0),
            conv_bn(256, 512, 2, 3),
        )
        self.layer8 = nn.Sequential(
            conv_bn(512, 128, 1, 1, 0),
            conv_bn(128, 256, 2, 3)
        )
        self.layer9 = nn.Sequential(
            conv_bn(256, 128, 1, 1, 0),
            conv_bn(128, 256, 2, 3)
        )
        self.layer10 = nn.Sequential(
            conv_bn(256, 64, 1, 1, 0),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        source_features = []
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        source_features.append(x)
        x = self.layer5(x)
        x = self.layer6(x)
        source_features.append(x)
        x = self.layer7(x)
        source_features.append(x)
        x = self.layer8(x)
        source_features.append(x)
        x = self.layer9(x)
        source_features.append(x)
        x = self.layer10(x)
        source_features.append(x)
        return source_features


def test_feature_extrator():
    sq = mobilenet_v1()
    #resnet = resnet_feature(BasicBlock, [2,2,2,2])
    
    inputs = torch.zeros((1,3, 300,300))
    inputs = Variable(inputs)
    features = sq.forward(inputs)
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
    



