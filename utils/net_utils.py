import torch
import torch.nn.init as init
import torch.nn as nn
def xavier(params):
    init.xavier_uniform_(params)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
