import torch
import torchvision

#from torchvision.datasets import CocoDetection
from coco import CocoDetection, AnnotationTransform
from utils.augmentations import SSDAugmentation
def make_dataset(dataset_name, dataroot, annFile = None, imageSize = 300):
    if dataset_name == 'coco':
        trans = torchvision.transforms
        transform = trans.Compose([trans.Resize((imageSize, imageSize)), trans.ToTensor(), trans.Normalize((0,0,0),(255,255,255))])
        ssd_dim = 300  # only support 300 now
        means = (0,0,0)
        transform = SSDAugmentation(ssd_dim, means)
        dataset = CocoDetection(dataroot, annFile, transform, AnnotationTransform())
        #data_iter = torch.utils.data.DataLoader(dataset, batch_size = 30, shuffle = True, num_workers = 1)
    return dataset
def test_coco_detect():
    dataset_name = 'coco_obj_detect'
    dataroot = '/datasets/coco/train2014/'
    annFile = '/datasets/coco/annotations/instances_train2014.json'
    dataset = make_dataset(dataset_name, dataroot, annFile )
    print(len(dataset))

if __name__ == '__main__':
    test_coco_detect()