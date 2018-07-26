import torch
import torchvision
import data
#from torchvision.datasets import CocoDetection
from .coco import CocoDetection, AnnotationTransform
from utils.augmentations import SSDAugmentation
from .voc0712 import VOCDetection 
def make_dataset(dataset_name, dataroot, annFile = None, means = None, imageSize = 300):
    if dataset_name == 'coco':
        trans = torchvision.transforms
        transform = trans.Compose([trans.Resize((imageSize, imageSize)), trans.ToTensor(), trans.Normalize((0,0,0),(255,255,255))])
        ssd_dim = 300  # only support 300 now
        means = (0,0,0)
        transform = SSDAugmentation(ssd_dim, means)
        dataset = CocoDetection(dataroot, annFile, transform, AnnotationTransform())
        #data_iter = torch.utils.data.DataLoader(dataset, batch_size = 30, shuffle = True, num_workers = 1)
    if dataset_name == 'voc':
        means = (104.0, 117.0, 123.0)
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(dataroot, train_sets, SSDAugmentation(
            imageSize, means), data.AnnotationTransform())
    return dataset
def test_coco_detect():
    dataset_name = 'coco_obj_detect'
    dataroot = '/datasets/coco/train2014/'
    annFile = '/datasets/coco/annotations/instances_train2014.json'
    dataset = make_dataset(dataset_name, dataroot, annFile )
    print(len(dataset))
def test_voc_dataset():
    pass
if __name__ == '__main__':
    test_coco_detect()