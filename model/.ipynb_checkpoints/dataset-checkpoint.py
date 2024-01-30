from torch.utils.data import Dataset
from netCDF4 import Dataset as netCDFDataset
import netCDF4
from PIL import Image
import numpy as np

class BlockingObservationalDataset(Dataset):
    def __init__(self):
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]
        self.data = netCDFDataset(
            "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc",
            mode="r",
        ).variables["z_0001"][:, :, :, 550:950]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label

class BlockingObservationalDataset1x1(Dataset):
    def __init__(self, transform=None):
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]
        self.data = netCDFDataset(
            "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc",
            mode="r",
        ).variables["z_0001"][:]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        images = [Image.fromarray(d) for d in data]
        
        if self.transform:
            images = [self.transform(i) for i in images]

        return np.array(images), label

class BlockingUKESMDataset1x1(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.labels = netCDFDataset(
            "./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc", mode="r"
        ).variables["blocking"][:]
        self.data = netCDFDataset(
            "./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc",
            mode="r",
        ).variables["z_0001"][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label