from torch.utils import data
import os
import numpy as np
from PIL import Image

class ImageNet(data.Dataset):

    def __init__(self, file_dir, transform=None):
        list_ = []
        for filename in os.listdir(file_dir):
             list_.append(os.path.join(file_dir, filename))

        self.dataName = list_
        self.transformation = transform

    def __len__(self):
        return len(self.dataName)

    def __getitem__(self, index):
        img = Image.open(self.dataName[index])
        if self.transformation is not None:

            data_ = self.transformation(img)

        return data_
