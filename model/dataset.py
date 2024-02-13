import netCDF4
from netCDF4 import Dataset as netCDFDataset
from torch.utils.data import Dataset
import xarray as xr
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
        xr_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc", mode="r")), decode_times=False)
        
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]

        # zero out negative values
        self.data = np.clip(xr_data.z_0001.data, 0, None)
        self.time = xr_data.time.data
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        return data, label, time
    
class SlpObservationalDataset(Dataset):
    def __init__(self, transform=None):
        xr_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/slp_final.nc", mode="r")), decode_times=False)
        
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]

        self.data = xr_data.msl.data
        self.time = xr_data.time.data
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        return data, label, time

class BlockingUKESMDataset1x1(Dataset):
    def __init__(self, transform=None, target_transform=None):
        xr_geo = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc", mode="r")), decode_times=False)

        self.labels = netCDFDataset(
            "./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc", mode="r"
        ).variables["blocking"][:]

        self.time = xr_geo.time.data

        # zero out negative values
        self.data = np.clip(xr_geo.z_0001.data, 0, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        return data, label, time

class GeopotentialSlpEra5Dataset(Dataset):
    def __init__(self, transform=None):
        xr_slp = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/slp_final.nc", mode="r")), decode_times=False)
        xr_geo = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc", mode="r")), decode_times=False)
        
        self.labels = netCDFDataset(
            "./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]

        self.time = xr_geo.time.data
        self.data = np.concatenate((np.clip(xr_geo.z_0001.data, 0, None), np.clip(xr_slp.msl.data, 0, None)), axis=1)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        return data, label, time
    
class GeopotentialEra5UkesmDataset(Dataset):
    def __init__(self, transform=None):
        xr_geo_era5 = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc", mode="r")), decode_times=False)
        xr_geo_ukesm = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc", mode="r")), decode_times=False)
        
        xr_era5_labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r")), decode_times=False)
        xr_ukesm_labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDFDataset("./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc", mode="r")), decode_times=False)
        self.labels = np.concatenate((xr_era5_labels.blocking.data, xr_ukesm_labels.blocking.data), axis=0)

        self.time = np.random.rand(len(xr_geo_ukesm.time.data) + len(xr_geo_era5.time.data))
        self.data = np.concatenate((xr_geo_era5.z_0001.data, xr_geo_ukesm.z_0001.data), axis=0)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        return data, label, time