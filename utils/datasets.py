from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image

class CIFAR100(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None, keyword='labeled'):
        self.root_dir = root_dir
        self.dset_list = pd.read_csv(csv_path, header=None).to_numpy()
        self.transforms = transforms

        self.labeled_idxs = [i for i, x in enumerate(self.dset_list) if x[1] == keyword]

        self.classes = sorted(os.listdir(self.root_dir))
        self.num_classes = len(self.classes)
        self.class_map = {}
        for i in range(len(self.classes)):
            self.class_map[self.classes[i]] = i

    def __len__(self):
        return len(self.labeled_idxs)
    
    def __getitem__(self,idx):

        dataset_idx = self.labeled_idxs[idx]
        self.img_name = self.dset_list[dataset_idx][0]
        self.img_path = os.path.join(self.root_dir, self.img_name)

        img = Image.open(self.img_path)

        if self.transforms:
            img = self.transforms(img)
        
        label = self.class_map[os.path.split(self.img_name)[0]]
        return {"image": img, "label": label,
                "name": self.img_name, "path": self.img_path}



class CIFAR100_BitLabeled(Dataset):
    def __init__(self, root_dir, guess_dataset, labeled_dataset, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.classes = sorted(os.listdir(self.root_dir))
        self.num_classes = len(self.classes)

        self.dataset = []
        for item in guess_dataset:
            self.dataset.append(item)
        
        for i in range(len(labeled_dataset)):
            img_dict = { 'image_path': labeled_dataset[i]['name'], 'label': labeled_dataset[i]['label']}
            self.dataset.append(img_dict)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):

        self.img_name = self.dataset[idx]['image_path']
        self.img_path = os.path.join(self.root_dir, self.img_name)

        img = Image.open(self.img_path)

        if self.transforms:
            img = self.transforms(img)
        
        label = self.dataset[idx]['label']
        if type(label) == list:
            label = - 1 * label[0]

        return {"image": img, "label": label,
                "name": self.img_name, "path": self.img_path}



