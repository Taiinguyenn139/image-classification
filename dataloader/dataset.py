import numpy as np
import pandas as pd
import torchvision
import os
import torch

from torch.utils.data import Dataset

class CLS_Dataset(Dataset):
    def __init__(self, data_path, pre_path, transform):
        super(CLS_Dataset).__init__()
        self.df = pd.read_csv(data_path)
        self.pre_path = pre_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = torchvision.io.read_image(os.path.join(self.pre_path,self.df["filepath"][index]))
        if img.size(dim=0)==1:
            img = torch.cat((img, img, img), 0)
        if self.transform:
            img = self.transform(img)
        label = self.df["label"][index]
        return img, label