import calc_BTD
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
import scipy
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage import transform
from skimage.segmentation import active_contour
from skimage.restoration import denoise_nl_means, estimate_sigma
from cv2 import cv2
from PIL import Image

# Land masking system
import rasterio
from pyresample import SwathDefinition, kd_tree
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer
from affine import Affine
from rasterio import mask
import geopandas

DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-sep12/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-sep12/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-aug08/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-aug08/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-may13/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-may13/"
TIFF_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24-tiff/"
LAND_POLYGON_SHAPE = "/Users/tschmidt/repos/tgs_honours/good_data/coastlines_merc/land_polygons.shp"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5

# Get projection info along with axis and fig objects for matplotlib
fig, ax, MapProj, FieldProj = matplot_consts.main_func()

# Get contents of data dir for ch 7
data_list_7 = os.listdir(DATA_DIR_7)
if ".DS_Store" in data_list_7:
    data_list_7.remove(".DS_Store") # For mac users
data_list_7 = sorted(data_list_7)

# Get contents of data dir for ch14
data_list_14 = os.listdir(DATA_DIR_14)
if ".DS_Store" in data_list_14:
    data_list_14.remove(".DS_Store") # For mac users
data_list_14 = sorted(data_list_14)

# Setup the sat constants used throughout the script.
first_ds_name = data_list_7[0]
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)
first_ds = Dataset(first_ds_path)
SatHeight = first_ds.variables['goes_imager_projection'].perspective_point_height
SatLon = first_ds.variables['goes_imager_projection'].longitude_of_projection_origin
SatSweep = first_ds.variables['goes_imager_projection'].sweep_angle_axis
X = first_ds.variables['x']
Y = first_ds.variables['y']
var_ch07 = first_ds.variables["Rad"]
first_ds = None # Free memory
var_ch07, lons, lats, extra = GOES.slice_sat_image(var_ch07, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch07 = np.where(lons==-999.99, np.nan, var_ch07) #UNEEDED???

###### New land masking system #######
tiff_path = os.path.join(TIFF_DIR, "0.tif")
height = var_ch07.shape[0]
width = var_ch07.shape[1]
p_crs = CRS.from_epsg(3857)
p_latlon = CRS.from_proj4("+proj=latlon")
crs_transform = Transformer.from_crs(p_latlon,p_crs)
ll_x, ll_y = crs_transform.transform(LLLon, LLLat)
ur_x, ur_y = crs_transform.transform(URLon, URLat)
area_extent = (ll_x, ll_y, ur_x, ur_y)
pixel_size_x = (ur_x - ll_x)/(width - 1)
pixel_size_y = (ur_y - ll_y)/(height - 1)
ul_x = ll_x # Why these?
ul_y = ur_y

area_id = "California Coast"
description = "See area ID"
proj_id = "Mercator"
swath_def = SwathDefinition(lons, lats)
new_affine = Affine(pixel_size_x, 0.0, ul_x, 0.0, -pixel_size_y, ul_y)
area_def = AreaDefinition(area_id, description, proj_id, p_crs,
                            width, height, area_extent) # TODO: Check to make sure that this is meant to be set to the target CRS

fill_value = -999.99 # Same as missing values for goes package
var_ch07_merc = kd_tree.resample_nearest(
    swath_def,
    var_ch07.ravel(),
    area_def,
    radius_of_influence=5000,
     nprocs=2,
    fill_value=fill_value
)
with rasterio.open(
    tiff_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1, #????
    dtype=var_ch07_merc.dtype,
    crs=p_crs,
    transform=new_affine,
    nodata=fill_value,
) as dst:
    dst.write(np.reshape(var_ch07_merc,(1,height,width))) #???

src = rasterio.open(tiff_path, mode='r+')
geodf = geopandas.read_file(LAND_POLYGON_SHAPE)
land_masking, other_affine = mask.mask(src, geodf[['geometry']].values.flatten(), invert=True, filled=False)
land_masking = np.ma.getmask(land_masking)
land_masking = np.reshape(land_masking, (height,width))
src.close() # Free memory
geodf = None # Free memory
#######################################

# if you want to use pcolormesh to plot data you will need calcute the corners of each pixel
loncor, latcor = GOES.get_lonlat_corners(lons, lats)


i = 0
for ds_name_7 in data_list_7:
    ds_name_14 = data_list_14[i]
    filename = str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    ds_path_7 = os.path.join(DATA_DIR_7, ds_name_7)
    ds_path_14 = os.path.join(DATA_DIR_14, ds_name_14)
    ds_7 = Dataset(ds_path_7)
    ds_14 = Dataset(ds_path_14)

    # Load channel 7
    X = ds_7.variables['x']
    Y = ds_7.variables['y']
    var_ch07 = ds_7.variables["Rad"]
    var_ch07, lons, lats, extra = GOES.slice_sat_image(var_ch07, X, Y, SatLon, SatHeight, SatSweep,
                                    LLLon, URLon, LLLat, URLat)
    var_ch07 = np.where(lons==-999.99, np.nan, var_ch07)

    # Load channel 14
    X = ds_14.variables['x']
    Y = ds_14.variables['y']
    var_ch14 = ds_14.variables["Rad"]
    var_ch14, lons, lats, extra = GOES.slice_sat_image(var_ch14, X, Y, SatLon, SatHeight, SatSweep,
                                    LLLon, URLon, LLLat, URLat)
    var_ch14 = np.where(lons==-999.99, np.nan, var_ch14)

    # Make BTD
    var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)
    var = kd_tree.resample_nearest(
        swath_def,
        var.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    var = np.where(var==-999.99, np.nan, var)
    var[land_masking] = np.nan

    # # Filter out the land
    # # (This is temporarilly hardcoded. Wil use shapefile later with rasterio)
    # lons = np.where(lons==-999.99, np.nan, lons)
    # lats = np.where(lats==-999.99, np.nan, lats)
    # var = np.where(lons > -118.5, np.nan, var)
    # var = np.where(np.logical_and(lons > -124.5, lats > 33.75), np.nan, var)

    # # Create mask array for the highest clouds
    # high_cloud_mask = calc_BTD.bt_ch14_temp_conv(var_ch14) < 5 # TODO: Make this more robust

    # #####TESTING######
    # # Use "golden arches" to filter out open ocean data
    # kernel_size = (3,3) # 2 or 3 seems optimal
    # BT = np.where(high_cloud_mask, np.nan, calc_BTD.bt_ch14_temp_conv(var_ch14)) # Remove highest clouds since they are the most irregular
    # BT_local_mean = scipy.ndimage.filters.generic_filter(BT, np.mean, kernel_size)
    # BT_local_SD = scipy.ndimage.filters.generic_filter(BT, np.std, kernel_size)

    # # plot.scatter_plt(BT_local_mean, BT_local_SD, "/Users/tschmidt/repos/tgs_honours/output/derp.png")

    # mean_cutoff = np.nanpercentile(BT_local_mean, 50)
    # SD_cutoff = np.nanpercentile(BT_local_SD, 95)

    # # mean_cutoff = (np.nanmax(BT_local_mean)-np.nanmin(BT_local_mean))*0.70 # Was 0.90
    # # SD_cutoff = (np.nanmax(BT_local_SD)-np.nanmin(BT_local_SD))*0.33 # Was 0.15
    # golden_arch_mask = np.logical_and(BT_local_mean > mean_cutoff, BT_local_SD < SD_cutoff)

    # # mean_cutoff = (np.nanmax(BT_local_mean)-np.nanmin(BT_local_mean))*0.65
    # # SD_cutoff = (np.nanmax(BT_local_SD)-np.nanmin(BT_local_SD))*0.15
    # # golden_arch_mask = np.logical_or(BT_local_mean > mean_cutoff, BT_local_SD > SD_cutoff)

    # var = np.where(golden_arch_mask, np.nan, var)
    # #################

    # #Filter out the cold high altitude clouds
    # var = np.where(high_cloud_mask, np.nan, var)

    plot.main_func(var, loncor, latcor, fig, ax, MapProj, FieldProj, file_path)

    print("Image " + str(i) + " Complete")
    i = i + 1