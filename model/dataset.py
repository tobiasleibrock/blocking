import os
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as netCDFDataset
import netCDF4

class BlockingObservationalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.labels = netCDFDataset('../data/GTD_1979-2019_JJAextd_8.nc', mode='r')
        self.data = netCDFDataset('../data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc', mode='r')

        # transform data and labels accordingly - should be done already pre-training
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return data, label