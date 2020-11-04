import plot
import matplot_consts

import GOES
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os

from cv2 import cv2

DATA_DIR_1 = "/Users/tschmidt/repos/tgs_honours/good_data/16-vis-apr24/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
GAMMA = 2.2
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5

# Get contents of data dir
data_list_1 = os.listdir(DATA_DIR_1)
if ".DS_Store" in data_list_1:
    data_list_1.remove(".DS_Store") # For mac users
data_list_1 = sorted(data_list_1)
first_ds_name = data_list_1.pop(0)
first_ds_path = os.path.join(DATA_DIR_1, first_ds_name)
first_ds = Dataset(first_ds_path)
SatHeight = first_ds.variables['goes_imager_projection'].perspective_point_height
SatLon = first_ds.variables['goes_imager_projection'].longitude_of_projection_origin
SatSweep = first_ds.variables['goes_imager_projection'].sweep_angle_axis
X = first_ds.variables['x']
Y = first_ds.variables['y']

var_ch01 = first_ds.variables["CMI_C01"]
var_ch01, lons, lats, extra = GOES.slice_sat_image(var_ch01, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch01 = np.where(lons==-999.99, np.nan, var_ch01)
var_ch01 = np.clip(var_ch01, 0, 1)
var_ch01 = np.power(var_ch01, 1/GAMMA)

var_ch02 = first_ds.variables["CMI_C02"]
var_ch02, lons, lats, extra = GOES.slice_sat_image(var_ch02, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch02 = np.where(lons==-999.99, np.nan, var_ch02)
var_ch02 = np.clip(var_ch02, 0, 1)
var_ch02 = np.power(var_ch02, 1/GAMMA)

var_ch03 = first_ds.variables["CMI_C03"]
var_ch03, lons, lats, extra = GOES.slice_sat_image(var_ch03, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch03 = np.where(lons==-999.99, np.nan, var_ch03)
var_ch03 = np.clip(var_ch03, 0, 1)
var_ch03 = np.power(var_ch03, 1/GAMMA)

#Make the missing "Green" channel
var_ch03 = 0.45*var_ch01 + 0.45*var_ch02 + 0.10*var_ch03 #From 2.2.1 in https://www.researchgate.net/publication/327401365_Generation_of_GOES-16_True_Color_Imagery_without_a_Green_Band/download
var_ch03 = np.clip(var_ch03, 0, 1)
rgb_image = np.dstack([var_ch02, var_ch03, var_ch01])

# Plot + save image
fig = plt.figure(dpi=150, figsize=(12, 9))
plt.imshow(rgb_image)#, vmin=VMIN, vmax=VMAX)
canvas = FigureCanvas(fig)
canvas.print_figure(os.path.join(OUT_DIR, 'vis.png'))

########################################################

# Get projection info along with axis and fig objects for matplotlib
fig, ax, MapProj, FieldProj = matplot_consts.main_func()
# Get contents of data dir for ch14
data_list_14 = os.listdir(DATA_DIR_14)
if ".DS_Store" in data_list_14:
    data_list_14.remove(".DS_Store") # For mac users
data_list_14 = sorted(data_list_14)
first_ds_name = data_list_14.pop(0)
first_ds_path = os.path.join(DATA_DIR_14, first_ds_name)
first_ds = Dataset(first_ds_path)
SatHeight = first_ds.variables['goes_imager_projection'].perspective_point_height
SatLon = first_ds.variables['goes_imager_projection'].longitude_of_projection_origin
SatSweep = first_ds.variables['goes_imager_projection'].sweep_angle_axis
X = first_ds.variables['x']
Y = first_ds.variables['y']
var_ch14 = first_ds.variables["Rad"]
var_ch14, lons, lats, extra = GOES.slice_sat_image(var_ch14, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch14 = np.where(lons==-999.99, np.nan, var_ch14)

# if you want to use pcolormesh to plot data you will need calcute the corners of each pixel
loncor, latcor = GOES.get_lonlat_corners(lons, lats)
plot.main_func(var_ch14, loncor, latcor, fig, ax, MapProj, FieldProj, os.path.join(OUT_DIR, 'ch_14.png'))