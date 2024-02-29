from re import L
import netCDF4
from netCDF4 import Dataset as netCDFDataset
from torch.utils.data import Dataset
import xarray as xr
import numpy as np


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        data, label, time = zip(*dataset)
        self.data = data
        self.labels = label
        self.time = time

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class GeoEra5Dataset5D(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_data = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:]

        self.labels = np.array(
            [
                1.0 if all(self.labels[i : i + 5]) else 0.0
                for i in range(len(self.labels) - 4)
            ]
            + [0] * 4
        )

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

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class GeoEra5Dataset(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_data = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
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

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class GeoEra5Dataset40(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_data = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_2019_beginAdjust_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
        ).variables["blocking"][:-98]

        # zero out negative values
        self.data = np.clip(xr_data.z_0001.data[:-98], 0, None)
        self.time = xr_data.time.data[:-98]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class SlpEra5Dataset(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_data = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(f"{prefix}./data/slp_era5_final.nc", mode="r")
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r"
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


class GeoUkesmDataset5D(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_geo = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc",
            mode="r",
        ).variables["blocking"][:]

        self.labels = np.array(
            [
                1.0 if all(self.labels[i : i + 5]) else 0.0
                for i in range(len(self.labels) - 4)
            ]
            + [0] * 4
        )

        self.time = xr_geo.time.data
        self.transform = transform

        # zero out negative values
        self.data = np.clip(xr_geo.z_0001.data, 0, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class GeoUkesmDataset(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_geo = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc",
            mode="r",
        ).variables["blocking"][:]

        self.time = xr_geo.time.data
        self.transform = transform

        # zero out negative values
        self.data = np.clip(xr_geo.z_0001.data, 0, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time


class GeoUkesmDataset100(Dataset):
    def __init__(self, transform=None, prefix=""):
        xr_geo = xr.open_dataset(
            xr.backends.NetCDF4DataStore(
                netCDFDataset(
                    f"{prefix}./data/500zg_day_UKESM1-0-LL_piControl_r1i1p1f2_gn_19600101-20601230_NHML_JJAextd_1x1_final.nc",
                    mode="r",
                )
            ),
            decode_times=False,
        )

        self.labels = netCDFDataset(
            f"{prefix}./data/labels/GTD_UKESM1-0-LL_piControl_1960-2060_JJAextd.nc",
            mode="r",
        ).variables["blocking"][:-98]

        self.time = xr_geo.time.data[:-98]
        self.transform = transform

        # zero out negative values
        self.data = np.clip(xr_geo.z_0001.data[:-98], 0, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        time = self.time[idx]

        if self.transform:
            data = np.transpose(data, (1, 2, 0))
            data = self.transform(image=data)["image"]
            data = np.transpose(data, (2, 0, 1))

        return data, label, time
