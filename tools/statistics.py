import os
import glob
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
## This script is for giving an overview of the dataset
TUNNLE_CLASSES = ['person', 'bicycle', 'car' , 'motorbike',
'aeroplane', 'bus', 'train' ,'truck', 'minibus', 'minitruck',
'suv','dangerouscar']

def class_distribution_info(dataroot, dataset = [1,2,3,4,6,7,8], save_dir = None, keep_difficult = False):
    num_classes = len(TUNNLE_CLASSES)
    dataset_info = pd.DataFrame(np.zeros((12, 2)).astype(np.int32),
                                columns = ['images', 'boxes'])
    dataset_info.index =  TUNNLE_CLASSES
    print(dataset_info.get_value('suv', 'boxes'))
    for set_num in dataset:
        images_folder = os.path.join(dataroot, str(set_num))
        for xml_path in glob.glob(os.path.join(images_folder, "*.xml")):
            img_id = xml_path.split('/')[-1].split(".")[0]
            target  = ET.parse(xml_path).getroot()
            for obj in target.iter('object'):
                difficult = int(obj.find('difficult').text) == 1
                if not keep_difficult and difficult:
                    continue
                name = obj.find('name').text.lower().strip()
                print(name)
                dataset_info.ix[name,'boxes'] +=1 
    print(dataset_info)

if __name__ == '__main__':
    import sys
    dataroot = sys.argv[1]
    class_distribution_info(dataroot)
    