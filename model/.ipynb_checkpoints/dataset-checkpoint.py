from torch.utils.data import Dataset
from netCDF4 import Dataset as netCDFDataset
import netCDF4

VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1

class BlockingObservationalDataset(Dataset):
    def __init__(self, run, transform=None, target_transform=None):
        #self.labels = netCDFDataset(
        #    "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        #)
        #self.data = netCDFDataset(
          #  "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc",
         #   mode="r",
        #)
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]
        self.data = netCDFDataset(
            "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc",
            mode="r",
        ).variables["z_0001"][:, :, :, 550:950]
        
        # transform data and labels accordingly - should be done already pre-training
        self.transform = transform
        self.target_transform = target_transform

        self.run = run

    def __len__(self):
        if self.run == "train":
            return int(len(self.data) * (1-VALIDATION_PERCENTAGE))
            #return int(len(self.data.variables["time"]) * (1-VALIDATION_PERCENTAGE))
        elif self.run == "val":
            #return int(len(self.data.variables["time"]) * VALIDATION_PERCENTAGE)
            return int(len(self.data) * VALIDATION_PERCENTAGE)

    def __getitem__(self, idx):
        #if self.run == "val": idx += int(len(self.data.variables["time"]) * (1-VALIDATION_PERCENTAGE))
        #data = self.data.variables["z_0001"][idx]
        #label = self.labels.variables["blocking"][idx]

        if self.run == "val": idx += int(len(self.data) * (1-VALIDATION_PERCENTAGE))
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label
