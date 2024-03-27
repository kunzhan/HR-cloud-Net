from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import argparse
 
class cn_landsat(Dataset):
    def __init__(self, name, root, mode, size=None, nsample=None):
        self.name = name
        self.id_path = '/idata/CH_Train/img.txt'
        # self.root = root
        self.root = '/idata/CH_Train'
        self.mini = '/data/Mini'
        self.mode = mode
        # crop size
        self.size = size
        self.val = "/idata/CH_Test/"
        val_path = '/idata/CH_Test/Test.txt'
        if mode == 'train_l' or mode == 'train_u':
            with open(self.id_path, 'r') as f:
            # with open(mini_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        # print(id)
        # CHL8
        if self.mode == 'train_l' or self.mode == 'train_u':
            # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            img = Image.open(os.path.join(self.root, 'img', id)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, 'mask', id )))/255)
        elif self.mode == 'test_mini':
            img = Image.open(os.path.join(self.mini, 'img', id)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.mini, 'mask', id )))/255)
            img, mask = normalize(img, mask)
            return img, mask, id
        else:
            img = Image.open(os.path.join(self.val, 'img', id)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.val, 'mask', id )))/255)
            img, mask = normalize(img, mask)
            return img, mask, id

        img_w, img_s = deepcopy(img), deepcopy(img)
        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        img_s = blur(img_s, p=0.5)


        # ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s = normalize(img_s)


        mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s, mask
    def __len__(self):
        return len(self.ids)