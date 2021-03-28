import calc_BTD

import GOES
from datetime import datetime, timedelta
import glob
import os
from pyresample import SwathDefinition, kd_tree
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer
import numpy as np
from cv2 import cv2

LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
DATA_DIR = "/Users/tschmidt/repos/tgs_honours/data/find_single_tracks/"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/ALL_DAYS_SUMMER_2019/"

fill_value = np.nan
area_id = "California Coast"
description = "See area ID"
proj_id = "Mercator"
p_crs = CRS.from_epsg(3857)
p_latlon = CRS.from_proj4("+proj=latlon")
crs_transform = Transformer.from_crs(p_latlon,p_crs)
ll_x, ll_y = crs_transform.transform(LLLon, LLLat)
ur_x, ur_y = crs_transform.transform(URLon, URLat)
area_extent = (ll_x, ll_y, ur_x, ur_y)

current_time = datetime.strptime("20190501", '%Y%m%d')
end_time = datetime.strptime("20190831", '%Y%m%d')

while current_time <= end_time:
    current_time_str_start = current_time.strftime("%Y%m%d") + "-190000"
    current_time_str_end = current_time.strftime("%Y%m%d") + "-190500"
    GOES.download('goes17', 'ABI-L1b-RadF', channel = ['07', '14'],
                DateTimeIni = current_time_str_start, DateTimeFin = current_time_str_end,
                rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR)

    data_list = os.listdir(DATA_DIR)
    if ".DS_Store" in data_list:
        data_list.remove(".DS_Store") # For mac users
    data_list = sorted(data_list)

    ds_path_7 = os.path.join(DATA_DIR, data_list[0])
    ds_path_14 = os.path.join(DATA_DIR, data_list[1])

    # Load channel 7
    ds_7 = GOES.open_dataset(ds_path_7)
    var_ch07, lons, lats = ds_7.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
    var_ch07, lons, lats = var_ch07.data, lons.data, lats.data
    HEIGHT = var_ch07.shape[0]
    WIDTH = var_ch07.shape[1]
    pixel_size_x = (ur_x - ll_x)/(WIDTH - 1)
    pixel_size_y = (ur_y - ll_y)/(HEIGHT - 1)
    area_def = AreaDefinition(area_id, description, proj_id, p_crs,
                                WIDTH, HEIGHT, area_extent)
    swath_def = SwathDefinition(lons, lats)
    var_ch07 = kd_tree.resample_nearest(
        swath_def,
        var_ch07.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )

    # Load channel 14
    ds_14 = GOES.open_dataset(ds_path_14)
    var_ch14, lons, lats = ds_14.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
    var_ch14, lons, lats = var_ch14.data, lons.data, lats.data
    swath_def = SwathDefinition(lons, lats)
    var_ch14 = kd_tree.resample_nearest(
        swath_def,
        var_ch14.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )

    # Make BTD
    var = calc_BTD.main_func_norm(var_ch14, var_ch07, 14, 7)

    # Plot it
    min_BTD = np.nanmin(var)
    if min_BTD < 0:
        var = var + np.abs(min_BTD)
    max_BTD = np.nanmax(var)
    var = var/max_BTD
    var = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)
    filename_BTD = current_time.strftime("%Y%m%d") + ".png"
    file_path_BTD = os.path.join(OUT_DIR, filename_BTD)
    cv2.imwrite(file_path_BTD, var)

    removal_path = os.path.join(DATA_DIR, "*")
    files = glob.glob(removal_path)
    for f in files:
        os.remove(f)

    current_time = current_time + timedelta(days=1)