import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.models.squeezenet import SqueezeNet
from torchvision.models.resnet import ResNet, BasicBlock

import sys
sys.path.append("../")
from layers import *
from data import v1,v3
import os

class SSD(nn.Module):
    def __init__(self, phase, base, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.base = base
        self.priorbox = PriorBox(v1)
        self.priors = Variable(self.priorbox.forward(), volatile = True)
        self.size = 300
        self.make_loc_and_conf()
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 3, 10, 0.01, 0.45)
    def make_loc_and_conf(self, out_channels = [128, 256, 512, 256,256], num_boxes = [4,6,6,6,6]):
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

def add_extras(in_channels, batch_norm = False):
    layers = []    
    flag = False
    for k, v in enumerate(cfg):
        pass
class resnet_feature(ResNet):
    def __init__(self, block, layers):
        super(resnet_feature,self).__init__(block, layers)
        self.layer5 = self._make_layer(block, 256, 2, stride = 2)
        self.layer6 = self._make_layer(block, 256, 2, stride = 2)
        self.layer7 = self._make_layer(block, 256, 2, stride = 2)
    def forward(self, x):
        source_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        source_features.append(x)
        x = self.layer3(x)
        source_features.append(x)
        x = self.layer4(x)
        source_features.append(x)
        x = self.layer5(x)
        source_features.append(x)
        x = self.layer6(x)
        source_features.append(x)
        return source_features
def test_feature_extrator():
    sq = SqueezeNet_feature()
    resnet = resnet_feature(BasicBlock, [2,2,2,2])
    inputs = torch.zeros((1,3, 300,300))
    inputs = Variable(inputs)
    features = resnet(inputs)

def test_SSD():
    resnet = resnet_feature(BasicBlock, [2,2,2,2])
    ssd_detector = SSD('train', resnet, 3)
    inputs = torch.zeros((1,3, 300,300))
    inputs = Variable(inputs)
    o = resnet.forward(inputs)
    for x in o:
        print(x.size())
if __name__ == '__main__':
    test_SSD()
    



