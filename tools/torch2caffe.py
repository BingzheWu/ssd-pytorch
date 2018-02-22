import torch
import sys
sys.path.append('/home/bingzhe/tools/pytorch2caffe')
#sys.path.append('/home/bingzhe/tools/nn_tools')
import torch.nn as nn
sys.path.append('../')
from torch.autograd import Variable
import models.mobilenet as mobilenet 
import torch.nn.init as init
import os
from pytorch2caffe import pytorch2caffe, plot_graph, pytorch2prototxt
from prototxt import *
#import pytorch_to_caffe as p2c
def xavier(param):
    init.xavier_uniform(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        #m.bias.data.zero_()

#model = mobilenet.SSD('train', mobilenet.mobilenet_v1(), 3)
model = mobilenet.mobilenet_v1()
model.apply(weights_init)
model.eval()
input_var = Variable(torch.rand(1, 3, 300, 300))
output_var = model(input_var)
output_dir = 'caffemodel_dir'

'''
# plot graph to png
#plot_graph(output_var, os.path.join(output_dir, 'mobilenet_v1.dot'))
pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'mv1-pytorch2caffe.prototxt'),
              os.path.join(output_dir, 'mv1-pytorch2caffe.caffemodel'))
'''
net_info = pytorch2prototxt(input_var, output_var)
print_prototxt(net_info)
#save_prototxt(net_info, os.path.join(output_dir, 'mv1.prototxt'))
