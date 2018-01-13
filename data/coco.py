import torch
import torchvision
from torch.utils import data
import os
from PIL import Image
import numpy as np
class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target:
            bbox = obj['bbox']
            class_id = obj['category_id']
            if class_id not in [1,17,18]:
                continue
            bndbox = []
            x,y,w,h = bbox
            x_max = x + w
            y_max = y + h
            x_max, x = x_max /float(width), x /float(width)
            y_max, y = y_max/float(height), y/float(height)
            bndbox = [x,y,x_max, y_max]
            if class_id == 1:
                label_idx = 0
            elif class_id == 17:
                label_idx = 1
            else:
                label_idx = 2
            bndbox.append(label_idx)
            res += [bndbox]
        return res 

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        catNms = ['person', 'dog', 'cat']
        self.ids = []
        for catNm in catNms:
            catIds = self.coco.getCatIds(catNms=catNm)
            imgIds = self.coco.getImgIds(catIds=catIds)
            self.ids = list(set(self.ids+imgIds))
        self.ids = list(self.ids)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        width, height = img.size
        img = np.array(img)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            target = target.astype(np.float32)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            #img = img[:, :, (2,1,0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2,0,1), target
    def __len__(self):
        return len(self.ids)
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def test_coco():
    coco_det = CocoDetection(root ='/home/bingzhe/datasets/coco/train2017',
        annFile = '/home/bingzhe/datasets/coco/annotations/instances_train2017.json',
        target_transform=AnnotationTransform())
    img, target = coco_det[29]
    print(img)
    print(target)
if __name__ == '__main__':
    test_coco()