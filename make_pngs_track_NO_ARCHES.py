import calc_BTD
import plot
import matplot_consts
from multi_tracker_improved import MultiTrackerImproved

# import GOES
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
import copy
from sklearn.cluster import KMeans

# Land masking system
import rasterio
from pyresample import SwathDefinition, kd_tree
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer, Proj
from affine import Affine
from rasterio import mask
import geopandas

# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch2-apr24/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-aug08-SHORT/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-aug08-SHORT/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch2-aug08-SHORT/"
DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-apr24/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-apr24/"
DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-apr24/"


TIFF_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24-tiff/"
LAND_POLYGON_SHAPE = "/Users/tschmidt/repos/tgs_honours/good_data/coastlines_merc/land_polygons.shp"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
HEIGHT = 1924 # These were determined from GOES library slice in old system
WIDTH = 2682 # TODO: TEST ROBUSTNESS OF THESE

# Get projection info along with axis and fig objects for matplotlib
fig, ax, fig2, ax2, MapProj, FieldProj = matplot_consts.main_func()

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

# Get contents of data dir for ch 2
data_list_2 = os.listdir(DATA_DIR_2)
if ".DS_Store" in data_list_2:
    data_list_2.remove(".DS_Store") # For mac users
data_list_2 = sorted(data_list_2)

# Make the projection object for the 500m shortwave band
first_ds_name = data_list_2[0]
first_ds_path = os.path.join(DATA_DIR_2, first_ds_name)
first_ds = Dataset(first_ds_path)
SatHeight = first_ds.variables['goes_imager_projection'].perspective_point_height # TODO: Decide if it is bad that I am reusing these same sat constants throught the bands and day
SatLon = first_ds.variables['goes_imager_projection'].longitude_of_projection_origin
SatSweep = first_ds.variables['goes_imager_projection'].sweep_angle_axis
ch02_goes_proj = Proj(proj='geos', h=SatHeight, lon_0=SatLon, sweep=SatSweep)

# Setup additional projection constants used throughout the script.
# TODO: When using the new latlon mesh pyproj system make sure it can handle missing values such as -999.99
# var_ch07 = np.where(lons==-999.99, np.nan, var_ch07) # THIS IS OLD WAY OF DOING THIS ^^^^^^^
tiff_path = os.path.join(TIFF_DIR, "0.tif")
p_crs = CRS.from_epsg(3857)
p_latlon = CRS.from_proj4("+proj=latlon")
crs_transform = Transformer.from_crs(p_latlon,p_crs)
ll_x, ll_y = crs_transform.transform(LLLon, LLLat)
ur_x, ur_y = crs_transform.transform(URLon, URLat)
area_extent = (ll_x, ll_y, ur_x, ur_y)
ul_x = ll_x # Why these?
ul_y = ur_y
area_id = "California Coast"
description = "See area ID"
proj_id = "Mercator"
pixel_size_x = (ur_x - ll_x)/(WIDTH - 1)
pixel_size_y = (ur_y - ll_y)/(HEIGHT - 1)
new_affine = Affine(pixel_size_x, 0.0, ul_x, 0.0, -pixel_size_y, ul_y)
area_def = AreaDefinition(area_id, description, proj_id, p_crs,
                            WIDTH, HEIGHT, area_extent)
fill_value = np.nan

# Make the projection object for the 2km longwave bands (Maybe make seperate for ch7 + ch14?)
first_ds_name = data_list_7[0]
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)
first_ds = Dataset(first_ds_path)
ch07_goes_proj = Proj(proj='geos', h=SatHeight, lon_0=SatLon, sweep=SatSweep)

# Load ch7 for land masking
X = first_ds.variables['x']
Y = first_ds.variables['y']
var_ch07 = first_ds.variables["Rad"][:]
X, Y = np.meshgrid(X*SatHeight, Y*SatHeight)
lons, lats = ch07_goes_proj(X, Y, inverse=True)
swath_def = SwathDefinition(lons, lats)
lons = None # Free the memory from these big datasets
lats = None
X = None
Y = None
first_ds = None
var_ch07 = kd_tree.resample_nearest(
    swath_def,
    var_ch07.ravel(),
    area_def,
    radius_of_influence=5000,
     nprocs=2,
    fill_value=fill_value
)
swath_def = None # Free the swath_def memory before the coastline interpolation

###### New land masking system #######################
with rasterio.open(
    tiff_path,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=1, #????
    dtype=var_ch07.dtype,
    crs=p_crs,
    transform=new_affine,
    nodata=fill_value,
) as dst:
    dst.write(np.reshape(var_ch07,(1,HEIGHT,WIDTH)))
var_ch07 = None # Free memory

src = rasterio.open(tiff_path, mode='r+')
geodf = geopandas.read_file(LAND_POLYGON_SHAPE)
land_masking, other_affine = mask.mask(src, geodf[['geometry']].values.flatten(), invert=True, filled=False)
land_masking = np.ma.getmask(land_masking)
land_masking = np.reshape(land_masking, (HEIGHT,WIDTH))
src.close() # Free memory
src = None
geodf = None
############################################################

# Init multi-tracker
trackers = MultiTrackerImproved(cv2.TrackerCSRT_create)

image_list = []
BTD_list = []
golden_arch_list = [] #TODO: Both of these next two are temp!!!
high_cloud_list = []

i = 0
for ds_name_7 in data_list_7:
    ds_name_14 = data_list_14[i]
    ds_name_2 = data_list_2[i]
    ds_path_7 = os.path.join(DATA_DIR_7, ds_name_7)
    ds_path_14 = os.path.join(DATA_DIR_14, ds_name_14)
    ds_path_2 = os.path.join(DATA_DIR_2, ds_name_2)
    
    # Load channel 2
    ds_2 = Dataset(ds_path_2)
    var_ch02 = ds_2.variables["Rad"][:]
    X = ds_2.variables['x']
    Y = ds_2.variables['y']
    ds_2 = None
    X, Y = np.meshgrid(X*SatHeight, Y*SatHeight)
    lons, lats = ch02_goes_proj(X, Y, inverse=True)
    X = None
    Y = None
    swath_def = SwathDefinition(lons, lats)
    lons = None
    lats = None
    var_ch02 = kd_tree.resample_nearest(
        swath_def,
        var_ch02.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    swath_def = None
    ds_2 = None

    # Load channel 7
    ds_7 = Dataset(ds_path_7)
    var_ch07 = ds_7.variables["Rad"][:]
    X = ds_7.variables['x']
    Y = ds_7.variables['y']
    ds_7 = None
    X, Y = np.meshgrid(X*SatHeight, Y*SatHeight)
    lons, lats = ch07_goes_proj(X, Y, inverse=True)
    X = None
    Y = None
    swath_def = SwathDefinition(lons, lats)
    lons = None
    lats = None
    var_ch07 = kd_tree.resample_nearest(
        swath_def,
        var_ch07.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    swath_def = None
    ds_7 = None

    # Load channel 14
    ds_14 = Dataset(ds_path_14)
    var_ch14 = ds_14.variables["Rad"][:]
    X = ds_14.variables['x']
    Y = ds_14.variables['y']
    ds_14 = None
    X, Y = np.meshgrid(X*SatHeight, Y*SatHeight)
    lons, lats = ch07_goes_proj(X, Y, inverse=True)
    X = None
    Y = None
    swath_def = SwathDefinition(lons, lats)
    lons = None
    lats = None
    var_ch14 = kd_tree.resample_nearest(
        swath_def,
        var_ch14.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    swath_def = None
    ds_14 = None

    # Make BTD
    var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)

    # Make copy of the BTD for use as a backround in cv2 image output
    # Maps the BTD values to a range of [0,255]
    BTD = copy.deepcopy(var) #TODO: Is the deepcopy unnecessary?
    BTD_img = copy.deepcopy(var) #TODO: Is the deepcopy unnecessary?
    min_BTD = np.nanmin(BTD_img)
    if min_BTD < 0:
        BTD_img = BTD_img + np.abs(min_BTD)
    max_BTD = np.nanmax(BTD_img)
    BTD_img = BTD_img/max_BTD
    BTD_img = cv2.cvtColor(BTD_img*255, cv2.COLOR_GRAY2BGR)

    # Filter out the land
    var[land_masking] = np.nan

    # Create mask array for the highest clouds
    high_cloud_mask = calc_BTD.bt_ch14_temp_conv(var_ch14) < 5 # TODO: Make this more robust

    #### Use reflectivity of channel 2 and BT of channel 14 to filter out open ocean data ###########
    BT = calc_BTD.bt_ch14_temp_conv(var_ch14)

    BT = BT[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land since golden arches works best when only over water
    var_ch02 = var_ch02[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land since golden arches works best when only over water

    BT_and_CH02 = np.vstack((BT, var_ch02)).T
    km = KMeans(n_clusters=3)
    kmeans_labels = km.fit_predict(BT_and_CH02)

    cluster_means = km.cluster_centers_[:, 1]
    golden_arch_mask_ocean = kmeans_labels == np.nanargmin(cluster_means)
    golden_arch_mask = np.zeros(var.shape, dtype=bool)
    golden_arch_mask[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] = golden_arch_mask_ocean

    # plot.scatter_plt(BT, var_ch02, kmeans_labels, km, fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/cluster_" + str(i) + ".png")
    # plot.hexbin(BT, var_ch02, fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/hex_" + str(i) + ".png")

    # plot.scatter_plt_no_cluster(BT, var_ch02, fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/cluster_" + str(i) + ".png")

    var = np.where(golden_arch_mask, np.nan, var)
    ###############################################################################################

    #Filter out the cold high altitude clouds
    var = np.where(high_cloud_mask, np.nan, var)

    ##### TESTING #####################
    # Make a copy of our var before canny is applied (BTD_complete is for the trackers)
    BTD_complete = copy.deepcopy(var) #TODO: Is the deepcopy unnecessary?
    min_BTD = np.nanmin(BTD_complete)
    if min_BTD < 0:
        BTD_complete = BTD_complete + np.abs(min_BTD)
    max_BTD = np.nanmax(BTD_complete)
    BTD_complete = BTD_complete/max_BTD
    BTD_complete = cv2.cvtColor(BTD_complete*255, cv2.COLOR_GRAY2BGR)
    BTD_complete = np.array(BTD_complete).astype('uint8') # Since it seems the trackers need images of type uint8
    #################################

    # Make Canny # TODO: Try adding more edges again?
    # 0.8, 3, 7 worked for hard case!!!!!!
    # var = feature.canny(var, sigma = 3.0, low_threshold = 0, high_threshold = 1) # Was 0.3, 3, 10 #But maybe try HT set to 8?
    var = feature.canny(var, sigma = 2.8, low_threshold = 0, high_threshold = 1.5)
    var = np.where(var == np.nan, 0, var)

    ## Skimage hough line transform #################################
    var = np.array(var).astype('uint8')
    img = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)

    # Was 0, 30, 1
    threshold = 0
    minLineLength = 30
    maxLineGap = 6
    theta = np.linspace(-np.pi, np.pi, 1000)

    lines = transform.probabilistic_hough_line(var, threshold=threshold, line_length=minLineLength, line_gap=maxLineGap, theta=theta)
    #############################################################

    #### TRACKER ################# #TODO: Try applying tracker to BTD_complete instead of canny edges (img)
    trackers.update(BTD_complete, i)

    if lines is not None:
        for line in lines:
            p0, p1 = line
            x1 = p0[0]
            y1 = p0[1]
            x2 = p1[0]
            y2 = p1[1]

            min_x = np.minimum(x1,x2)
            min_y = np.minimum(y1,y2)
            max_x = np.maximum(x1,x2)
            max_y = np.maximum(y1,y2)

            rect = (min_x-2, min_y-2, max_x-min_x + 4, max_y-min_y + 4) #TODO: Maybe expand the size of the boxes a bit?
            trackers.add_tracker(BTD_complete, rect, len(data_list_7))
    ###############################

    # Make line plots
    if lines is not None:
        for line in lines:
            p0, p1 = line
            x1 = p0[0]
            y1 = p0[1]
            x2 = p1[0]
            y2 = p1[1]
            # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    ####TEMP######################
    filename = "early_" + str(i) + "_.png"
    file_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(file_path, BTD_img)
    filename = "canny_" + str(i) + "_.png"
    file_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(file_path, img)
    ###############################
    
    image_list.append(BTD_img)
    BTD_list.append(BTD)
    golden_arch_list.append(golden_arch_mask)
    high_cloud_list.append(high_cloud_mask)

    print("Image " + str(i) + " Calculated")
    i = i + 1


for i in range(len(data_list_7)):
    filename = str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    boxes = trackers.get_boxes(i)

    BTD_img = image_list[i]
    BTD = BTD_list[i]

    # Make box plots for trackers
    # Also make and highlight the labels
    labels = np.zeros([BTD.shape[0], BTD.shape[1], 3], dtype=np.float32)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]

        if w > 0 and h > 0 and x >= 0 and y >= 0 and y+h <= BTD.shape[0] and x+w <= BTD.shape[1] and y < BTD.shape[0] and x < BTD.shape[1]:
            box_slice = BTD[y:y+h, x:x+w]
            labels_slice = labels[y:y+h, x:x+w, 2]
            labels_slice = np.where(box_slice >= np.nanmax(box_slice)-2, 255.0, labels_slice)
            labels[y:y+h, x:x+w, 2] = labels_slice # Add red for labels

            # cv2.rectangle(BTD_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    ####TESTING######
    labels_slice = np.zeros([BTD.shape[0], BTD.shape[1]])
    labels_slice = np.where(golden_arch_list[i], 255.0, labels_slice)
    # labels_slice = np.where(high_cloud_list[i], 255.0, labels_slice)
    labels[:,:,2] = labels_slice
    ################

    BTD_img = cv2.addWeighted(BTD_img, 1.0, labels, 0.5, 0)

    cv2.imwrite(file_path, BTD_img)

    print("Image " + str(i) + " Complete")