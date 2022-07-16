import numpy as np
import os
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import scipy.io
import warnings
import random
from easydict import EasyDict as edict

from dataloader import base
from misc import camera

from dataloader.pascal3d import Dataset as Pascal3D
from dataloader.pix3d import Dataset as Pix3D

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        super().__init__(opt,split)
        self.cat = dict(
            car="car",
            chair="chair",
            plane="aeroplane"
        )[opt.data.pascal3d.cat]
        self.iteratable_list = []

        if split=='train':
            self.pascal3d_dataset = Pascal3D(opt, split)
            for iter in range(len(self.pascal3d_dataset.list)):
                self.iteratable_list.append({'dataset': 'pascal3d', 'idx': iter})

            self.pix3d_dataset = Pix3D(opt, split)
            for iter in range(len(self.pix3d_dataset.list)):
                self.iteratable_list.append({'dataset': 'pix3d', 'idx': iter})

            random.shuffle(self.iteratable_list)
        
        else:
            self.pix3d_dataset = Pix3D(opt, split)
            for iter in range(len(self.pix3d_dataset.list)):
                self.iteratable_list.append({'dataset': 'pix3d', 'idx': iter})        


    def __getitem__(self,idx):
        if self.iteratable_list[idx]['dataset']=='pix3d':
            return self.pix3d_dataset.__getitem__(self.iteratable_list[idx]['idx'])
        elif self.iteratable_list[idx]['dataset']=='pascal3d':
            return self.pascal3d_dataset.__getitem__(self.iteratable_list[idx]['idx'])

    def __len__(self):
        return len(self.iteratable_list)
    
