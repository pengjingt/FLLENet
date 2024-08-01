import os
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random



class CULane(Dataset):
    def __init__(self, path, image_set='train', transforms=None):
        super(CULane, self).__init__()
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()

    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))

        self.img_list = []
        # self.segLabel_list = []
        # self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
        return self.img_list

    def createIndex_test(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))

        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, line[1:]))
        return self.img_list


class lowlight_loader(Dataset):

    def __init__(self, path):
        self.path = path
        culane_set = CULane(self.path)
        self.train_list = culane_set.img_list
        self.size = (288, 828)

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        data_lowlight = data_lowlight.resize(self.size, Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


