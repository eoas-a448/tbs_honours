import calc_BTD
import plot
import matplot_consts
from multi_tracker_improved import MultiTrackerImproved

import GOES
import numpy as np
import os
from skimage import feature
from skimage import transform
from cv2 import cv2
import copy

# New clustering method
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import DBSCAN

# Land masking system
import rasterio
from pyresample import SwathDefinition, kd_tree
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer, Proj
from affine import Affine
from rasterio import mask
import geopandas

class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def decision_function(self, X):
        return self.classifier_.decision_function(X)

# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-jul10/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-jul10/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-jul10/"
# DATA_DIR_SIZE = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-jul10/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-jul10/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-jul03/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-jul03/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-jul03/"
# DATA_DIR_SIZE = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-jul03/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-jul03/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-aug08/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-aug08/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-aug08/"
# DATA_DIR_SIZE = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-aug08/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-aug08/"
DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-apr24/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-apr24/"
DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-apr24/"
DATA_DIR_SIZE = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-apr24/"
DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-apr24/"
DATA_DIR_CLEAR = "/Users/tschmidt/repos/tgs_honours/good_data/17-clear_sky-apr24/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-jun02/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-jun02/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-jun02/"
# DATA_DIR_SIZE = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-jun02/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-jun02/"

TIFF_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24-tiff/"
LAND_POLYGON_SHAPE = "/Users/tschmidt/repos/tgs_honours/good_data/coastlines_merc/land_polygons.shp"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5

# Get projection info along with axis and fig objects for matplotlib
fig, ax, fig2, ax2, fig3, ax3, MapProj, FieldProj = matplot_consts.main_func()

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

# Get contents of data dir for cloud particle size
data_list_size = os.listdir(DATA_DIR_SIZE)
if ".DS_Store" in data_list_size:
    data_list_size.remove(".DS_Store") # For mac users
data_list_size = sorted(data_list_size)

# Get contents of data dir for cloud optical depth
data_list_depth = os.listdir(DATA_DIR_DEPTH)
if ".DS_Store" in data_list_depth:
    data_list_depth.remove(".DS_Store") # For mac users
data_list_depth = sorted(data_list_depth)

# Get contents of data dir for given clear sky mask
data_list_clear = os.listdir(DATA_DIR_CLEAR)
if ".DS_Store" in data_list_clear:
    data_list_clear.remove(".DS_Store") # For mac users
data_list_clear = sorted(data_list_clear)

# Load ch7 for projection constants
first_ds_name = data_list_7[0]
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)
first_ds = GOES.open_dataset(first_ds_path)
var_ch02, lons, lats = first_ds.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
var_ch02, lons, lats = var_ch02.data, lons.data, lats.data
HEIGHT = var_ch02.shape[0]
WIDTH = var_ch02.shape[1]

# Setup projection constants used throughout the script.
# TODO: When using the new GOES slicing system make sure it can handle missing values such as -999.99
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

# Load ch7 for land masking
first_ds_name = data_list_7[0]
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)
first_ds = GOES.open_dataset(first_ds_path)
var_ch07, lons, lats = first_ds.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
var_ch07, lons, lats = var_ch07.data, lons.data, lats.data
swath_def = SwathDefinition(lons, lats)
first_ds = None # Free the memory from these big datasets
var_ch07 = kd_tree.resample_nearest(
    swath_def,
    var_ch07.ravel(),
    area_def,
    radius_of_influence=5000,
     nprocs=2,
    fill_value=fill_value
)

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

src = rasterio.open(tiff_path, mode='r+')
geodf = geopandas.read_file(LAND_POLYGON_SHAPE)
land_masking, other_affine = mask.mask(src, geodf[['geometry']].values.flatten(), invert=True, filled=False)
land_masking = np.ma.getmask(land_masking)
land_masking = np.reshape(land_masking, (HEIGHT,WIDTH))
src.close() # Free memory
src = None
geodf = None
############################################################

# Make blue land highlight
land_labels = np.zeros([var_ch07.shape[0], var_ch07.shape[1], 3], dtype=np.float32)
land_labels[:,:,0] = np.where(land_masking, 255.0, land_labels[:,:,0])
land_labels_canny = np.zeros([var_ch07.shape[0], var_ch07.shape[1], 3], dtype=np.uint8)
land_labels_canny[:,:,0] = np.where(land_masking, 255, land_labels_canny[:,:,0])

# Init multi-tracker
trackers = MultiTrackerImproved(cv2.TrackerCSRT_create)

image_list = []
ship_track_mask_list = []

i = 0
for ds_name_7 in data_list_7:
    ds_name_14 = data_list_14[i]
    ds_name_2 = data_list_2[i]
    ds_name_size = data_list_size[i]
    ds_name_depth = data_list_depth[i]
    ds_path_7 = os.path.join(DATA_DIR_7, ds_name_7)
    ds_path_14 = os.path.join(DATA_DIR_14, ds_name_14)
    ds_path_2 = os.path.join(DATA_DIR_2, ds_name_2)
    ds_path_size = os.path.join(DATA_DIR_SIZE, ds_name_size)
    ds_path_depth = os.path.join(DATA_DIR_DEPTH, ds_name_depth)
    ds_name_clear = data_list_clear[i]
    ds_path_clear = os.path.join(DATA_DIR_CLEAR, ds_name_clear)

    # Load channel 2
    ds_2 = GOES.open_dataset(ds_path_2)
    var_ch02, lons, lats = ds_2.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
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

    # Load channel 7
    ds_7 = GOES.open_dataset(ds_path_7)
    var_ch07, lons, lats = ds_7.image("Rad", domain=[LLLon, URLon, LLLat, URLat])
    var_ch07, lons, lats = var_ch07.data, lons.data, lats.data
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

    # Load cloud particle size
    ds_size = GOES.open_dataset(ds_path_size)
    var_size, lons, lats = ds_size.image("PSD", domain=[LLLon, URLon, LLLat, URLat])
    var_size, lons, lats = var_size.data, lons.data, lats.data
    swath_def = SwathDefinition(lons, lats)
    var_size = kd_tree.resample_nearest(
        swath_def,
        var_size.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )

    # Load cloud optical depth
    ds_depth = GOES.open_dataset(ds_path_depth)
    var_depth, lons, lats = ds_depth.image("COD", domain=[LLLon, URLon, LLLat, URLat])
    var_depth, lons, lats = var_depth.data, lons.data, lats.data
    swath_def = SwathDefinition(lons, lats)
    var_depth = kd_tree.resample_nearest(
        swath_def,
        var_depth.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )

    # Load given open ocean mask
    ds_clear = GOES.open_dataset(ds_path_clear)
    var_clear, lons, lats = ds_clear.image("BCM", domain=[LLLon, URLon, LLLat, URLat])
    var_clear, lons, lats = var_clear.data, lons.data, lats.data
    swath_def = SwathDefinition(lons, lats)
    var_clear = kd_tree.resample_nearest(
        swath_def,
        var_clear.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    

    # Make BTD
    var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)

    # Skip frame if it has bad data
    if np.isnan(var).any():
        i = i + 1
        continue

    # # Output true open ocean labels
    # given_open_ocean = np.zeros([var_clear.shape[0], var_clear.shape[1]], dtype=np.float32)
    # given_open_ocean = np.where(np.logical_and(var_clear == 0, np.logical_not(land_masking)), 1.0, given_open_ocean)
    # filename = "true_ocean_" + str(i) + ".tif"
    # file_path = os.path.join(OUT_DIR, filename)
    # with rasterio.open(
    #     file_path,
    #     "w",
    #     driver="GTiff",
    #     height=HEIGHT,
    #     width=WIDTH,
    #     count=1, #????
    #     dtype=given_open_ocean.dtype,
    #     crs=p_crs,
    #     transform=new_affine,
    #     nodata=fill_value,
    # ) as dst:
    #     dst.write(np.reshape(given_open_ocean,(1,HEIGHT,WIDTH)))

    # Make copy of the BTD for use as a backround in cv2 image output
    # Maps the BTD values to a range of [0,255]
    BTD_img = copy.deepcopy(var)
    min_BTD = np.nanmin(BTD_img)
    if min_BTD < 0:
        BTD_img = BTD_img + np.abs(min_BTD)
    max_BTD = np.nanmax(BTD_img)
    BTD_img = BTD_img/max_BTD
    BTD_img = cv2.cvtColor(BTD_img*255, cv2.COLOR_GRAY2BGR)
    BTD_img_trackers = copy.deepcopy(BTD_img) # Next two lines are for new BTD data for trackers
    BTD_img_trackers = np.array(BTD_img_trackers).astype('uint8') # Since it seems the trackers need images of type uint8

    # Filter out the land
    var[land_masking] = np.nan
    var_size[land_masking] = np.nan
    var_depth[land_masking] = np.nan

    # Create mask array for the highest clouds
    high_cloud_mask = calc_BTD.bt_ch14_temp_conv(var_ch14) < 5 # TODO: Make this more robust

    #### Use reflectivity of channel 2 and BT of channel 14 to filter out open ocean data ############
    BT = calc_BTD.bt_ch14_temp_conv(var_ch14)

    BT = BT[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land
    var_ch02 = var_ch02[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land

    BT_and_CH02 = np.vstack((BT, var_ch02)).T
    BT_and_CH02_sample, _ = train_test_split(BT_and_CH02, train_size=10000)

    clusterer = DBSCAN(eps=1.5, min_samples=100)
    classifier = DecisionTreeClassifier()
    inductive_cluster = InductiveClusterer(clusterer, classifier).fit(BT_and_CH02_sample)
    IC_labels = inductive_cluster.predict(BT_and_CH02) + 1
    
    all_labels = np.unique(IC_labels)
    min_refl = np.Inf
    open_ocean_label = 0
    for j in all_labels:
        labeled_refl_array = var_ch02[IC_labels==j]
        mean_refl = np.nanmean(labeled_refl_array)
        if mean_refl < min_refl:
            open_ocean_label = j
            min_refl = mean_refl
    open_ocean_mask_1D = IC_labels == open_ocean_label

    open_ocean_mask = np.zeros(var.shape, dtype=bool)
    open_ocean_mask[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] = open_ocean_mask_1D

    # plot.scatter_plt_2_cluster_legend(BT[IC_labels != open_ocean_label], var_ch02[IC_labels != open_ocean_label], BT[IC_labels == open_ocean_label], var_ch02[IC_labels == open_ocean_label], fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/cluster_" + str(i) + ".png")
    # plot.hexbin(BT, var_ch02, i, fig3, ax3, "/Users/tschmidt/repos/tgs_honours/output/hex_" + str(i) + ".png")

    var = np.where(open_ocean_mask, np.nan, var)

    # # Output generated ocean labels
    # open_ocean = np.zeros([var_clear.shape[0], var_clear.shape[1]], dtype=np.float32)
    # open_ocean = np.where(open_ocean_mask, 1.0, open_ocean)
    # filename = "ocean_" + str(i) + ".tif"
    # file_path = os.path.join(OUT_DIR, filename)
    # with rasterio.open(
    #     file_path,
    #     "w",
    #     driver="GTiff",
    #     height=HEIGHT,
    #     width=WIDTH,
    #     count=1, #????
    #     dtype=open_ocean.dtype,
    #     crs=p_crs,
    #     transform=new_affine,
    #     nodata=fill_value,
    # ) as dst:
    #     dst.write(np.reshape(open_ocean,(1,HEIGHT,WIDTH)))
    ###############################################################################################

    # Filter out the cold high altitude clouds
    var = np.where(high_cloud_mask, np.nan, var)
    # var_size = np.where(high_cloud_mask, np.nan, var_size)
    # var_depth = np.where(high_cloud_mask, np.nan, var_depth)

    # Make Canny ###########################################################
    # 0.8, 3, 7 worked for hard case!!!!!!
    # var = feature.canny(var, sigma = 3.0, low_threshold = 0, high_threshold = 1) # Was 0.3, 3, 10 #But maybe try HT set to 8?
    # 2.5, 0, 0.8 jul 03
    var = feature.canny(var, sigma = 2.5, low_threshold = 0, high_threshold = 0.8)
    var = np.where(var == np.nan, 0, var)
    ########################################################################

    ## Skimage hough line transform #################################
    var = np.array(var).astype('uint8')
    img = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)

    # Was 0, 30, 1
    # NOW 0, 15, 1
    threshold = 0
    minLineLength = 40
    maxLineGap = 3
    theta = np.linspace(-np.pi, np.pi, 1000)

    lines = transform.probabilistic_hough_line(var, threshold=threshold, line_length=minLineLength, line_gap=maxLineGap, theta=theta)
    #############################################################

    #### TRACKER #################
    # TODO: Use BTD_img_trackers that is made above instead of canny (img)???????????
    trackers.update(img, i)

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
            trackers.add_tracker(img, rect, len(data_list_7))
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

    # Add blue land highlight
    img = cv2.addWeighted(img, 1.0, land_labels_canny, 1.0, 0)

    # Find ship track pixels ###################################################################################
    # def phil_eq(depth):
    #     return 7 + depth**(3/10)

    # August 08
    def phil_eq(depth):
        return 6 + depth**(5/10)

    # July 03
    # def phil_eq(depth):
    #     return 8 + depth**(3/10)

    size_func = phil_eq(var_depth)

    # ship_mask = np.logical_and(np.logical_and(var_size > size_func-1, var_size < size_func + 1), np.logical_and(var_size < 20, var_depth < 60))
    # ship_mask = np.logical_and(np.logical_and(var_size > 0, var_size < size_func), np.logical_and(np.logical_not(np.isnan(var_size)), np.logical_and(var_size < 20, var_depth < 30)))
    # ship_mask = np.logical_and(np.logical_and(var_size > size_func-0.3, var_size < size_func + 0.3), np.logical_and(np.logical_not(np.isnan(var_size)), np.logical_and(var_size < 20, var_depth < 30)))
    ship_mask = np.logical_and(np.logical_and(var_size > 0, var_size < size_func), np.logical_not(np.isnan(var_size))) # August 08
    # ship_mask = np.logical_and(np.logical_and(var_size > size_func - 3, var_size < size_func), np.logical_not(np.isnan(var_size)))
    # ship_mask = np.logical_and(np.logical_and(var_size > 0, var_size < size_func), np.logical_and(np.logical_not(np.isnan(var_size)), np.logical_and(var_depth > 0, var_depth < 1000))) # Jul 03
    # ship_mask = np.logical_and(np.logical_and(var_size > 0, var_size < 11), np.logical_not(np.isnan(var_size)))
    ############################################################################################################

    ####TEMP#########################################
    filename = "canny_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(file_path, img)

    # Make BTD with blue highlighted land
    BTD_img2 = copy.deepcopy(BTD_img)
    BTD_img2 = cv2.addWeighted(BTD_img2, 1.0, land_labels, 1.0, 0)
    filename = "BTD_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(file_path, BTD_img2)

    # BTD_img2 = copy.deepcopy(BTD_img)
    filename = "NEW_MASK_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    labels_slice = np.zeros([BTD_img.shape[0], BTD_img.shape[1]])
    labels_slice = np.where(ship_mask, 255.0, labels_slice)
    labels = np.zeros([BTD_img.shape[0], BTD_img.shape[1], 3], dtype=np.float32)
    labels[:,:,2] = labels_slice
    BTD_img2 = cv2.addWeighted(BTD_img2, 1.0, labels, 0.5, 0)
    cv2.imwrite(file_path, BTD_img2)

    # filename = "NEW_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin_log(var_depth, var_size, i, fig3, ax3, file_path)

    # filename = "NEW_CLUSTER_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt_log(var_depth[np.logical_not(ship_mask)], var_size[np.logical_not(ship_mask)], var_depth[ship_mask], var_size[ship_mask], fig2, ax2, file_path)

    # min_size = np.nanmin(var_size)
    # if min_size < 0:
    #     var_size = var_size + np.abs(min_size)
    # max_size = np.nanmax(var_size)
    # var_size = var_size/max_size
    # var_size = cv2.cvtColor(var_size*255, cv2.COLOR_GRAY2BGR)
    # filename_size = "cloud_size_" + str(i) + ".png"
    # file_path_size = os.path.join(OUT_DIR, filename_size)
    # cv2.imwrite(file_path_size, var_size)

    # min_depth = np.nanmin(var_depth)
    # if min_depth < 0:
    #     var_depth = var_depth + np.abs(min_depth)
    # max_depth = np.nanmax(var_depth)
    # var_depth = var_depth/max_depth
    # var_depth = cv2.cvtColor(var_depth*255, cv2.COLOR_GRAY2BGR)
    # filename_depth = "cloud_depth_" + str(i) + ".png"
    # file_path_depth = os.path.join(OUT_DIR, filename_depth)
    # cv2.imwrite(file_path_depth, var_depth)
    ################################################################
    
    image_list.append(BTD_img)
    ship_track_mask_list.append(ship_mask)

    print("Image " + str(i) + " Calculated")
    i = i + 1


for i in range(len(image_list)):
    filename = str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    boxes = trackers.get_boxes(i)

    BTD_img = image_list[i]
    ship_mask = ship_track_mask_list[i]

    # Make box plots for trackers
    # Also make and highlight the labels
    labels = np.zeros([BTD_img.shape[0], BTD_img.shape[1], 3], dtype=np.float32)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]

        if w > 0 and h > 0 and x >= 0 and y >= 0 and y+h <= BTD_img.shape[0] and x+w <= BTD_img.shape[1] and y < BTD_img.shape[0] and x < BTD_img.shape[1]:
            box_slice = ship_mask[y:y+h, x:x+w]

            labels_slice = labels[y:y+h, x:x+w, 2]
            labels_slice = np.where(box_slice, 255.0, labels_slice)
            labels[y:y+h, x:x+w, 2] = labels_slice # Add red for labels

            # cv2.rectangle(BTD_img, (x, y), (x + w, y + h), (0, 255.0, 0), 2)

    # Add blue land highlight
    BTD_img = cv2.addWeighted(BTD_img, 1.0, land_labels, 1.0, 0)

    # Add all ship track labels
    BTD_img = cv2.addWeighted(BTD_img, 1.0, labels, 0.5, 0)

    cv2.imwrite(file_path, BTD_img)

    print("Image " + str(i) + " Complete")