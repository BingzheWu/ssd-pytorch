import torch
import sys
sys.path.append('/home/bingzhe/projects/ssd.pytorch/')
sys.path.append('/home/bingzhe/nn_tools')
from models.model_factory import build_ssd
import pytorch_to_caffe

def torch2onnx(model_file):
    net = build_ssd('export', 300, 8, 'mobilenet_v1')
    net.load_weights(model_file)
    net.eval()
    x = torch.randn(1, 3, 300, 300)
    torch_out = torch.onnx._export(net, x, "mobilenetv1_8classes.onnx", export_params = True)
def torch2caffe(model_file):
    net = build_ssd('export', 300, 8, 'mobilenet_v1')
    net.load_weights(model_file)
    net.eval()
    x = torch.randn(1, 3, 300, 300)
    pytorch_to_caffe.trans_net(net, x,'mbv1')
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format('mbv1'))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format('mbv1'))
if __name__ == '__main__':
    import sys
    model_file = sys.argv[1]
    torch2caffe(model_file)
