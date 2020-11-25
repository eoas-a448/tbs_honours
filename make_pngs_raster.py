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
import rasterio
from rasterio import mask
import rioxarray
import xarray
import geopandas
from shapely.geometry import mapping
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage import transform
from skimage.segmentation import active_contour
from skimage.restoration import denoise_nl_means, estimate_sigma
from cv2 import cv2
from PIL import Image

DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-sep12/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-sep12/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-aug08/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-aug08/"
TIFF_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24-tiff/"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
COASTLINE_SHP = "/Users/tschmidt/repos/tgs_honours/good_data/coastlines/land_polygons.shp"
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
first_ds_name = data_list_7.pop(0)
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)

####RASTERIO####
kernel_size = 8
xds = xarray.open_dataset(first_ds_path)

first_tiff_path = os.path.join(TIFF_DIR, "test.tif")
# xds["Rad"].rio.to_raster(first_tiff_path)
SatHeight = xds['goes_imager_projection'].perspective_point_height
SatLon = xds['goes_imager_projection'].longitude_of_projection_origin
SatSweep = xds['goes_imager_projection'].sweep_angle_axis
X = xds['x']
Y = xds['y']
var_ch07 = xds["Rad"]
var_ch07, lons, lats, extra = GOES.slice_sat_image(var_ch07, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)

# xds = xds.drop_vars(["x","y","Rad","DQF"])
# xds['x'] = ('x', X[514:1185])
# xds['y'] = ('y', Y[883:1364])
# xds = xds.rio.set_spatial_dims('x','y') # Uneeded???
# xds["Rad"] = (('y','x'), var_ch07)

xds = xds.sel(y=slice(*[Y[883],Y[1363]]),x=slice(*[X[514],X[1184]]))
# xds = xds.assign_coords({"lons": (("y","x"), np.where(lons==-999.99, np.nan, lons))}) #TODO: REMOVE -999s
# xds = xds.assign_coords({"lats": (("y","x"),  np.where(lats==-999.99, np.nan, lats))}) #TODO: REMOVE -999s

xds["lons"] = (("y","x"), np.where(lons==-999.99, np.nan, lons)) #????
xds["lats"] = (("y","x"), np.where(lons==-999.99, np.nan, lons))

xds = xds.rio.set_crs(rasterio.crs.CRS.from_string("epsg:3857")) #TODO: REMOVE THIS HARDCODING
rad_data = xds["Rad"]
rad_data = rad_data.rio.set_crs(rasterio.crs.CRS.from_string("epsg:3857")) #TODO: REMOVE THIS HARDCODING
rad_data.rio.to_raster(first_tiff_path)
src = rasterio.open(first_tiff_path, mode='r+')

geodf = geopandas.read_file("/Users/tschmidt/repos/tgs_honours/good_data/coastlines/land_polygons.shp")

out = mask.mask(src, geodf[['geometry']].values.flatten())

print(out)

# scipy.ndimage.filters.generic_filter(data, np.nanmean, size = kernel_size)
############

first_ds = Dataset(first_ds_path)
SatHeight = first_ds.variables['goes_imager_projection'].perspective_point_height
SatLon = first_ds.variables['goes_imager_projection'].longitude_of_projection_origin
SatSweep = first_ds.variables['goes_imager_projection'].sweep_angle_axis
X = first_ds.variables['x']
Y = first_ds.variables['y']
var_ch07 = first_ds.variables["Rad"]
var_ch07, lons, lats, extra = GOES.slice_sat_image(var_ch07, X, Y, SatLon, SatHeight, SatSweep,
                                        LLLon, URLon, LLLat, URLat)
var_ch07 = np.where(lons==-999.99, np.nan, var_ch07)

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

# Make BTD
var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)

# Make Canny
var = np.where(calc_BTD.bt_ch14_temp_conv(var_ch14) < 5, np.nan, var)
var = np.where(var < 2, np.nan, var)
# var = np.where(var > 18, np.nan, var)
var = feature.canny(var, sigma = 0.3, low_threshold = 3, high_threshold = 10) # High threshold between 8-10

plot.main_func(var, loncor, latcor, fig, ax, MapProj, FieldProj, os.path.join(OUT_DIR, '0.png'))

i = 1
for ds_name_7 in data_list_7:
    ds_name_14 = data_list_14[i-1]
    filename = str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    ds_path_7 = os.path.join(DATA_DIR_7, ds_name_7)
    ds_path_14 = os.path.join(DATA_DIR_14, ds_name_14)
    ds_7 = Dataset(ds_path_7)
    ds_14 = Dataset(ds_path_14)

    X = ds_7.variables['x']
    Y = ds_7.variables['y']
    var_ch07 = ds_7.variables["Rad"]
    var_ch07, lons, lats, extra = GOES.slice_sat_image(var_ch07, X, Y, SatLon, SatHeight, SatSweep,
                                    LLLon, URLon, LLLat, URLat)
    var_ch07 = np.where(lons==-999.99, np.nan, var_ch07)

    X = ds_14.variables['x']
    Y = ds_14.variables['y']
    var_ch14 = ds_14.variables["Rad"]
    var_ch14, lons, lats, extra = GOES.slice_sat_image(var_ch14, X, Y, SatLon, SatHeight, SatSweep,
                                    LLLon, URLon, LLLat, URLat)
    var_ch14 = np.where(lons==-999.99, np.nan, var_ch14)

    # Make BTD
    var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)

    #####TESTING######
    # # Use "golden arches" to filter out open ocean data
    # kernel_size = (8,8)
    # scipy.ndimage.filters.generic_filter()
    #################

    # Make Canny
    var = np.where(calc_BTD.bt_ch14_temp_conv(var_ch14) < 5, np.nan, var)
    # var = np.where(var < 2, np.nan, var)
    var = feature.canny(var, sigma = 0.3, low_threshold = 3, high_threshold = 10) # Was 0.3, 3, 10 #But maybe try HT set to 8?
    var = np.where(var == np.nan, 0, var)

    #####TESTING########
    # # var = np.where(var < 2, np.nan, var)
    # var = filters.hessian(var, gamma=15)
    # var = np.where(calc_BTD.bt_ch14_temp_conv(var_ch14) < 5, 0, var)
    # var = np.where(var < 1, 0, var)
    ####################

    #########TESTING#######
    # # Hough line transform
    # var = np.array(var).astype('uint8')
    # img = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)

    # rho = 100
    # theta = np.pi/2
    # threshold = 10
    # minLineLength = 20
    # maxLineGap = 0

    # lines = cv2.HoughLinesP(var, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)
    # if lines is not None:
    #     N = lines.shape[0]
    #     for j in range(N):
    #         x1 = lines[j][0][0]
    #         y1 = lines[j][0][1]
    #         x2 = lines[j][0][2]
    #         y2 = lines[j][0][3]
    #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    # cv2.imwrite(file_path, img)
    #######################

    ########TESTING#########
    # Skimage hough line transform
    var = np.array(var).astype('uint8')
    img = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)

    threshold = 0
    minLineLength = 30
    maxLineGap = 1
    theta = np.linspace(-np.pi, np.pi, 1000)

    lines = transform.probabilistic_hough_line(var, threshold=threshold, line_length=minLineLength, line_gap=maxLineGap, theta=theta)

    if lines is not None:
        for line in lines:
            p0, p1 = line
            x1 = p0[0]
            y1 = p0[1]
            x2 = p1[0]
            y2 = p1[1]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite(file_path, img)
    ########################

    # plot.main_func(var, loncor, latcor, fig, ax, MapProj, FieldProj, file_path)

    print("Image " + str(i) + " Complete")
    i = i + 1