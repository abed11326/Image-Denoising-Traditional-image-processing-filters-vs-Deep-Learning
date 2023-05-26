import cv2
import os
import numpy as np

def get_noised(img):
    noise = np.random.randn(256, 256, 3) * 0.05 
    noised_img = img + noise
    noised_img = np.clip(noised_img, 0, 1)
    return noised_img

data_path = './Data/original'
noised_path = './Data/noised'

parts = ['train', 'val']
labels = os.listdir(os.path.join(data_path, 'train'))

for part in parts:
    for label in labels:
        path = os.path.join(data_path, part, label)
        images = os.listdir(path)
        os.makedirs(os.path.join(noised_path, part, label), exist_ok=True)

        for image in images:
            img_path = os.path.join(path, image)
            img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) / 255.0
            noised = get_noised(img)
            #cv2.imwrite(img_path, img * 255)
            cv2.imwrite(os.path.join(noised_path, part, label, image), noised * 255)