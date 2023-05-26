import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import hflip
import numpy as np

class Data(Dataset):
    def __init__(self, data_path, labels_path=None, training=True):
        super(Data, self).__init__()
        self.data = []
        self.labels = []
        self.label_img = labels_path
        self.training = training
        data_path = os.path.join(data_path, 'train' if training else 'val')
        labels_path = os.path.join(labels_path, 'train' if training else 'val') if labels_path else None
        for cls in os.listdir(data_path):
            for image in os.listdir(os.path.join(data_path, cls)):
                self.data.append(os.path.join(data_path, cls, image))
                if labels_path:
                    label = os.path.join(labels_path, cls, image)
                else:
                    label = torch.zeros(20)
                    label[int(cls)] = 1.0
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_im = read_image(self.data[index]) / 255.0
        output = (read_image(self.labels[index]) / 255.0) if self.label_img else self.labels[index]
        if self.training:
            if np.random.rand() < 0.5:
                input_im = hflip(input_im)
                output = hflip(output) if self.label_img else output
        return input_im, output
