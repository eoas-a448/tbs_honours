from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import GOES
import numpy as np
from pyresample import SwathDefinition, kd_tree
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer
import os

from cv2 import cv2

DATA_DIR_1 = "/Users/tschmidt/repos/tgs_honours/good_data/17-vis-apr24/"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
GAMMA = 2.2
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
fill_value = np.nan

# Get contents of data dir
data_list_1 = os.listdir(DATA_DIR_1)
if ".DS_Store" in data_list_1:
    data_list_1.remove(".DS_Store") # For mac users
data_list_1 = sorted(data_list_1)
first_ds_name = data_list_1.pop(0)
first_ds_path = os.path.join(DATA_DIR_1, first_ds_name)

first_ds = GOES.open_dataset(first_ds_path)
var_ch01, lons, lats = first_ds.image("CMI_C01", domain=[LLLon, URLon, LLLat, URLat])
var_ch01, lons, lats = var_ch01.data, lons.data, lats.data
swath_def = SwathDefinition(lons, lats)

HEIGHT = var_ch01.shape[0]
WIDTH = var_ch01.shape[1]
p_crs = CRS.from_epsg(3857)
p_latlon = CRS.from_proj4("+proj=latlon")
crs_transform = Transformer.from_crs(p_latlon,p_crs)
ll_x, ll_y = crs_transform.transform(LLLon, LLLat)
ur_x, ur_y = crs_transform.transform(URLon, URLat)
area_extent = (ll_x, ll_y, ur_x, ur_y)
area_id = "California Coast"
description = "See area ID"
proj_id = "Mercator"
area_def = AreaDefinition(area_id, description, proj_id, p_crs,
                            WIDTH, HEIGHT, area_extent)

var_ch01 = kd_tree.resample_nearest(
    swath_def,
    var_ch01.ravel(),
    area_def,
    radius_of_influence=5000,
    nprocs=2,
    fill_value=fill_value
)
var_ch01 = np.clip(var_ch01, 0, 1)
var_ch01 = np.power(var_ch01, 1/GAMMA)

var_ch02, lons, lats = first_ds.image("CMI_C02", domain=[LLLon, URLon, LLLat, URLat])
var_ch02, lons, lats = var_ch02.data, lons.data, lats.data
swath_def = SwathDefinition(lons, lats)

var_ch02 = kd_tree.resample_nearest(
    swath_def,
    var_ch02.ravel(),
    area_def,
    radius_of_influence=5000,
    nprocs=2,
    fill_value=fill_value
)
var_ch02 = np.clip(var_ch02, 0, 1)
var_ch02 = np.power(var_ch02, 1/GAMMA)

var_ch03, lons, lats = first_ds.image("CMI_C03", domain=[LLLon, URLon, LLLat, URLat])
var_ch03, lons, lats = var_ch03.data, lons.data, lats.data
swath_def = SwathDefinition(lons, lats)

var_ch03 = kd_tree.resample_nearest(
    swath_def,
    var_ch03.ravel(),
    area_def,
    radius_of_influence=5000,
    nprocs=2,
    fill_value=fill_value
)
var_ch03 = np.clip(var_ch03, 0, 1)
var_ch03 = np.power(var_ch03, 1/GAMMA)

#Make the missing "Green" channel
var_ch03 = 0.45*var_ch01 + 0.45*var_ch02 + 0.10*var_ch03 #From 2.2.1 in https://www.researchgate.net/publication/327401365_Generation_of_GOES-16_True_Color_Imagery_without_a_Green_Band/download
var_ch03 = np.clip(var_ch03, 0, 1)

rgb_image = np.dstack([var_ch02, var_ch03, var_ch01])

fig = plt.figure(dpi=150, figsize=(12, 9))
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.imshow(rgb_image)
canvas = FigureCanvas(fig)
canvas.print_figure(os.path.join(OUT_DIR, 'vis.png'), bbox_inches='tight')