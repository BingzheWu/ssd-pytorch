import glob
import cv2
import os
dim = 300
raw_dir = '/home/bingzhe/datasets/otureo_car_detect/training_dataset/'
dst_dir = '/home/bingzhe/datasets/otureo_car_detect/training_300x300/'
def resize(image):
    image_out = cv2.resize(image, (dim, dim))
    return image_out

def preprocess_save(process_sets = [5,6,7,8]):
    for training_set in process_sets:
        images_dir = os.path.join(raw_dir, str(training_set))
        for image_file in glob.glob(os.path.join(images_dir, '*.png')):
            img_id = image_file.split('/')[-1]
            save_path = os.path.join(dst_dir, str(training_set), img_id)
            try:
                img = cv2.imread(image_file)
                img = resize(img)
                cv2.imwrite(save_path, img)
            except:
                continue
if __name__ == '__main__':
    preprocess_save()
