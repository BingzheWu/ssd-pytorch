import torch
import cv2
import time
import glob
import os
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def predict_one_image(net, image, transform = None):
    height, width = image.shape[:2]
    x = torch.from_numpy(transform(image)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0)
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
            cv2.putText(image, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                    2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return image
def save_detections(net, image_folder, save_folder):
    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))
    for image_file in glob.glob(os.path.join(image_folder, '*.png')):
        image_id = image_file.split('/')[-1]
        save_path = os.path.join(save_folder, image_id)
        try:
            image = cv2.imread(image_file)
            image = predict_one_image(net, image, transform)
            cv2.imwrite(save_path, image)
        except:
            print(image_file)
if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append('/home/bingzhe/projects/ssd.pytorch')
    from data import BaseTransform
    from data.tunnle_car import CLASSES as labelmap
    from models import model_factory
    image_folder = sys.argv[1]
    save_folder = sys.argv[2]
    trained_model = sys.argv[3]
    net = model_factory.build_ssd('test', 300, 8, 'mobilenet_v1')
    net.load_state_dict(torch.load(trained_model))
    save_detections(net, image_folder, save_folder)