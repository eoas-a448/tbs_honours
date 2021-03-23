import calc_BTD
import plot
import matplot_consts
from multi_tracker_improved import MultiTrackerImproved

import GOES
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage import transform
from skimage.segmentation import active_contour
from skimage.restoration import denoise_nl_means, estimate_sigma
from cv2 import cv2
import copy
from sklearn.cluster import KMeans

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

# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-aug08/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-aug08/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-aug08/"
# DATA_DIR_6 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch6-aug08/"
# DATA_DIR_AER = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-aug08/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-aug08/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-apr24/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-apr24/"
# DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-apr24/"
# DATA_DIR_6 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch6-apr24/"
# DATA_DIR_AER = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-apr24/"
# DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-apr24/"
# DATA_DIR_CLEAR = "/Users/tschmidt/repos/tgs_honours/good_data/17-clear_sky-apr24/"
DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch7-jun02/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-jun02/"
DATA_DIR_2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-jun02/"
DATA_DIR_6 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch6-jun02/"
DATA_DIR_AER = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_part_size-jun02/"
DATA_DIR_DEPTH = "/Users/tschmidt/repos/tgs_honours/good_data/17-cloud_depth-jun02/"
DATA_DIR_CLEAR = "/Users/tschmidt/repos/tgs_honours/good_data/17-clear_sky-jun02/"

TIFF_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24-tiff/"
LAND_POLYGON_SHAPE = "/Users/tschmidt/repos/tgs_honours/good_data/coastlines_merc/land_polygons.shp"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
# HEIGHT = 1924 # These were determined from GOES library slice in old system
# WIDTH = 2682 # TODO: TEST ROBUSTNESS OF THESE

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

# Get contents of data dir for ch 6
data_list_6 = os.listdir(DATA_DIR_6)
if ".DS_Store" in data_list_6:
    data_list_6.remove(".DS_Store") # For mac users
data_list_6 = sorted(data_list_6)

# TEMP #########################
data_list_aer = os.listdir(DATA_DIR_AER)
if ".DS_Store" in data_list_aer:
    data_list_aer.remove(".DS_Store") # For mac users
data_list_aer = sorted(data_list_aer)

data_list_depth = os.listdir(DATA_DIR_DEPTH)
if ".DS_Store" in data_list_depth:
    data_list_depth.remove(".DS_Store") # For mac users
data_list_depth = sorted(data_list_depth)

data_list_clear = os.listdir(DATA_DIR_CLEAR)
if ".DS_Store" in data_list_clear:
    data_list_clear.remove(".DS_Store") # For mac users
data_list_clear = sorted(data_list_clear)
##################################

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

# Init multi-tracker
trackers = MultiTrackerImproved(cv2.TrackerCSRT_create)

image_list = []
BTD_list = []
refl_ch2_list = []
refl_ch6_list = []
golden_arch_list = [] #TODO: Both of these next two are temp!!!
high_cloud_list = []
aer_list = []

i = 0
for ds_name_7 in data_list_7:
    ds_name_14 = data_list_14[i]
    ds_name_2 = data_list_2[i]
    ds_name_6 = data_list_6[i]
    ds_path_7 = os.path.join(DATA_DIR_7, ds_name_7)
    ds_path_14 = os.path.join(DATA_DIR_14, ds_name_14)
    ds_path_2 = os.path.join(DATA_DIR_2, ds_name_2)
    ds_path_6 = os.path.join(DATA_DIR_6, ds_name_6)

    # TEMP ##################################
    ds_name_aer = data_list_aer[i]
    da_path_aer = os.path.join(DATA_DIR_AER, ds_name_aer)
    ds_name_depth = data_list_depth[i]
    da_path_depth = os.path.join(DATA_DIR_DEPTH, ds_name_depth)
    ds_name_clear = data_list_clear[i]
    da_path_clear = os.path.join(DATA_DIR_CLEAR, ds_name_clear)
    ##########################################

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

    # Load channel 2 reflectivity
    ds_2 = GOES.open_dataset(ds_path_2)
    refl_var_ch02, lons, lats = ds_2.image("Rad", up_level=True, domain=[LLLon, URLon, LLLat, URLat])
    refl_var_ch02 = refl_var_ch02.refl_fact_to_refl(lons, lats).data
    swath_def = SwathDefinition(lons.data, lats.data)
    refl_var_ch02 = kd_tree.resample_nearest(
        swath_def,
        refl_var_ch02.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )

    # Load channel 6 reflectivity
    ds_6 = GOES.open_dataset(ds_path_6)
    refl_var_ch06, lons, lats = ds_6.image("Rad", up_level=True, domain=[LLLon, URLon, LLLat, URLat])
    refl_var_ch06 = refl_var_ch06.refl_fact_to_refl(lons, lats).data
    swath_def = SwathDefinition(lons.data, lats.data)
    refl_var_ch06 = kd_tree.resample_nearest(
        swath_def,
        refl_var_ch06.ravel(),
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

    # TEMP ###############################################################################
    ds_aer = GOES.open_dataset(da_path_aer)
    var_aer, lons, lats = ds_aer.image("PSD", domain=[LLLon, URLon, LLLat, URLat])
    var_aer, lons, lats = var_aer.data, lons.data, lats.data
    swath_def = SwathDefinition(lons, lats)
    var_aer = kd_tree.resample_nearest(
        swath_def,
        var_aer.ravel(),
        area_def,
        radius_of_influence=5000,
        nprocs=2,
        fill_value=fill_value
    )
    ds_depth = GOES.open_dataset(da_path_depth)
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
    ds_clear = GOES.open_dataset(da_path_clear)
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
    # min_aer = np.nanmin(var_clear)
    # if min_aer < 0:
    #     var_clear = var_clear + np.abs(min_aer)
    # max_aer = np.nanmax(var_clear)
    # var_clear = var_clear/max_aer
    # var_clear = cv2.cvtColor(var_clear*255, cv2.COLOR_GRAY2BGR)
    # filename_aer = "clear_sky_" + str(i) + ".png"
    # file_path_aer = os.path.join(OUT_DIR, filename_aer)
    # cv2.imwrite(file_path_aer, var_clear)

    # var_aer, lons, lats = ds_aer.image("Smoke", domain=[LLLon, URLon, LLLat, URLat])
    # var_aer, lons, lats = var_aer.data, lons.data, lats.data
    # swath_def = SwathDefinition(lons, lats)
    # var_aer = kd_tree.resample_nearest(
    #     swath_def,
    #     var_aer.ravel(),
    #     area_def,
    #     radius_of_influence=5000,
    #     nprocs=2,
    #     fill_value=fill_value
    # )
    # min_aer = np.nanmin(var_aer)
    # if min_aer < 0:
    #     var_aer = var_aer + np.abs(min_aer)
    # max_aer = np.nanmax(var_aer)
    # var_aer = var_aer/max_aer
    # var_aer = cv2.cvtColor(var_aer*255, cv2.COLOR_GRAY2BGR)
    # filename_aer = "smoke_" + str(i) + ".png"
    # file_path_aer = os.path.join(OUT_DIR, filename_aer)
    # cv2.imwrite(file_path_aer, var_aer)

    # var_aer, lons, lats = ds_aer.image("Dust", domain=[LLLon, URLon, LLLat, URLat])
    # var_aer, lons, lats = var_aer.data, lons.data, lats.data
    # swath_def = SwathDefinition(lons, lats)
    # var_aer = kd_tree.resample_nearest(
    #     swath_def,
    #     var_aer.ravel(),
    #     area_def,
    #     radius_of_influence=5000,
    #     nprocs=2,
    #     fill_value=fill_value
    # )
    # min_aer = np.nanmin(var_aer)
    # if min_aer < 0:
    #     var_aer = var_aer + np.abs(min_aer)
    # max_aer = np.nanmax(var_aer)
    # var_aer = var_aer/max_aer
    # var_aer = cv2.cvtColor(var_aer*255, cv2.COLOR_GRAY2BGR)
    # filename_aer = "dust_" + str(i) + ".png"
    # file_path_aer = os.path.join(OUT_DIR, filename_aer)
    # cv2.imwrite(file_path_aer, var_aer)
    ########################################################################################

    # Make BTD
    var = calc_BTD.main_func(var_ch14, var_ch07, 14, 7)

    # Skip day if it has bad data
    if np.isnan(var).any():
        i = i + 1
        continue

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
    BTD_img_trackers = copy.deepcopy(BTD_img) # Next two lines are for new BTD data for trackers
    BTD_img_trackers = np.array(BTD_img_trackers).astype('uint8') # Since it seems the trackers need images of type uint8

    # Filter out the land
    var[land_masking] = np.nan

    # Create mask array for the highest clouds
    high_cloud_mask = calc_BTD.bt_ch14_temp_conv(var_ch14) < 5 # TODO: Make this more robust

    #### Use reflectivity of channel 2 and BT of channel 14 to filter out open ocean data ###########
    BT = calc_BTD.bt_ch14_temp_conv(var_ch14)

    BT = BT[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land since golden arches works best when only over water
    var_ch02 = var_ch02[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] # Filter out the land since golden arches works best when only over water

    BT_and_CH02 = np.vstack((BT, var_ch02)).T
    BT_and_CH02_sample, _ = train_test_split(BT_and_CH02, train_size=10000)

    ######DELETE########
    # clusterer = OPTICS(min_cluster_size=0.1, xi=0.035)#, cluster_method="dbscan", max_eps=8)
    ###################
    clusterer = DBSCAN(eps=1.5, min_samples=100)
    classifier = DecisionTreeClassifier()
    inductive_cluster = InductiveClusterer(clusterer, classifier).fit(BT_and_CH02_sample)
    kmeans_labels = inductive_cluster.predict(BT_and_CH02) + 1

    # km = KMeans(n_clusters=3)
    # kmeans_labels = km.fit_predict(BT_and_CH02)
    # cluster_means = km.cluster_centers_[:, 1]
    # golden_arch_mask_ocean = kmeans_labels == np.nanargmin(cluster_means)
    
    all_labels = np.unique(kmeans_labels)
    min_refl = np.Inf
    open_ocean_label = 0
    for j in all_labels:
        labeled_refl_array = var_ch02[kmeans_labels==j]
        mean_refl = np.nanmean(labeled_refl_array)
        if mean_refl < min_refl:
            open_ocean_label = j
            min_refl = mean_refl
    golden_arch_mask_ocean = kmeans_labels == open_ocean_label

    golden_arch_mask = np.zeros(var.shape, dtype=bool)
    golden_arch_mask[np.logical_and(np.logical_not(land_masking), np.logical_not(high_cloud_mask))] = golden_arch_mask_ocean

    km = "test" ####TEMP
    # plot.scatter_plt(BT, var_ch02, kmeans_labels, km, fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/cluster_" + str(i) + ".png")
    # plot.hexbin(BT, var_ch02, fig2, ax2, "/Users/tschmidt/repos/tgs_honours/output/hex_" + str(i) + ".png")

    var = np.where(golden_arch_mask, np.nan, var)
    ###############################################################################################

    #Filter out the cold high altitude clouds
    var = np.where(high_cloud_mask, np.nan, var)

    # ##### TESTING #####################
    # # Make a copy of our var before canny is applied (BTD_complete is for the trackers)
    # BTD_complete = copy.deepcopy(var) #TODO: Is the deepcopy unnecessary?
    # min_BTD = np.nanmin(BTD_complete)
    # if min_BTD < 0:
    #     BTD_complete = BTD_complete + np.abs(min_BTD)
    # max_BTD = np.nanmax(BTD_complete)
    # BTD_complete = BTD_complete/max_BTD
    # BTD_complete = cv2.cvtColor(BTD_complete*255, cv2.COLOR_GRAY2BGR)
    # BTD_complete = np.array(BTD_complete).astype('uint8') # Since it seems the trackers need images of type uint8
    # #################################

    # Make Canny # TODO: Try adding more edges again?
    # 0.8, 3, 7 worked for hard case!!!!!!
    # var = feature.canny(var, sigma = 3.0, low_threshold = 0, high_threshold = 1) # Was 0.3, 3, 10 #But maybe try HT set to 8?
    # var = feature.canny(var, sigma = 2.7, low_threshold = 0, high_threshold = 1.2)
    var = feature.canny(var, sigma = 2.2, low_threshold = 0, high_threshold = 1.2)
    var = np.where(var == np.nan, 0, var)

    ## Skimage hough line transform #################################
    var = np.array(var).astype('uint8')
    img = cv2.cvtColor(var*255, cv2.COLOR_GRAY2BGR)

    # Was 0, 30, 1
    threshold = 0
    minLineLength = 15
    maxLineGap = 1
    theta = np.linspace(-np.pi, np.pi, 1000)

    lines = transform.probabilistic_hough_line(var, threshold=threshold, line_length=minLineLength, line_gap=maxLineGap, theta=theta)
    #############################################################

    #### TRACKER ################# #TODO: Try applying tracker to BTD_complete instead of canny edges (img)
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

    # # Make line plots
    # if lines is not None:
    #     for line in lines:
    #         p0, p1 = line
    #         x1 = p0[0]
    #         y1 = p0[1]
    #         x2 = p1[0]
    #         y2 = p1[1]
    #         cv2.line(BTD_img,(x1,y1),(x2,y2),(0,255,0),2)


    # Experimental ship track particle clustering #############################
    var_depth[land_masking] = np.nan
    var_aer[land_masking] = np.nan
    var_depth[high_cloud_mask] = np.nan
    var_aer[high_cloud_mask] = np.nan
    var_depth[var_clear == 0] = np.nan
    var_aer[var_clear == 0] = np.nan
    # var_depth[golden_arch_mask] = np.nan
    # var_aer[golden_arch_mask] = np.nan

    size_mask = var_aer < 15
    # var_depth = var_depth[size_mask]
    # var_aer = var_aer[size_mask]

    BTD_max = np.nanmax(var_depth)
    error_cluster_mask = var_depth < BTD_max - 5
    BTD_small = var_depth[np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(var_aer)),np.logical_not(np.isnan(var_depth))),error_cluster_mask),size_mask)]
    var_aer_small = var_aer[np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(var_aer)),np.logical_not(np.isnan(var_depth))),error_cluster_mask),size_mask)]

    BTD_small = np.log(BTD_small)
    var_aer_small = np.log(var_aer_small)

    var_aer_max = np.max(var_aer_small)
    var_aer_small = var_aer_small/var_aer_max
    BTD_min = np.min(BTD_small)
    if BTD_min < 0:
        BTD_small = BTD_small + np.abs(BTD_min)
    BTD_max = np.max(BTD_small)
    BTD_small = BTD_small/BTD_max

    # error_cluster_mask = BTD_small < (BTD_max-5)/BTD_max
    # BTD_small = BTD_small[error_cluster_mask]
    # var_aer_small = var_aer_small[error_cluster_mask]

    var_aer_and_BTD = np.vstack((BTD_small, var_aer_small)).T
    var_aer_and_BTD_sample, _ = train_test_split(var_aer_and_BTD, train_size=10000)

    clusterer = DBSCAN(eps=0.016, min_samples=100) # Found through extensive testing
    classifier = DecisionTreeClassifier()
    inductive_cluster = InductiveClusterer(clusterer, classifier).fit(var_aer_and_BTD_sample)
    IC_labels = inductive_cluster.predict(var_aer_and_BTD) + 1
    
    all_labels = np.unique(IC_labels)
    max_particle_depth = 0
    ship_particle_label = 0
    for j in all_labels:
        if j == 0: # Skip outliers
            continue
        labeled_particle_array = BTD_small[IC_labels==j]
        mean_depth = np.nanmean(labeled_particle_array)
        if mean_depth > max_particle_depth:
            ship_particle_label = j
            max_particle_depth = mean_depth

    BTD_small = BTD_small*BTD_max
    if BTD_min < 0:
        BTD_small = BTD_small - np.abs(BTD_min)
    var_aer_small = var_aer_small*var_aer_max

    BTD_small = np.exp(BTD_small)
    var_aer_small = np.exp(var_aer_small)
    
    # mean_depth = np.nanmean(BTD_small[IC_labels == ship_particle_label])
    # mean_size = np.nanmean(var_aer_small[IC_labels == ship_particle_label])

    # ship_particle_label = 0
    # biggest_cluster = 0
    # for j in all_labels:
    #     if j == 0: # Skip outliers
    #         continue
    #     cluster_size = np.sum(IC_labels==j)
    #     if cluster_size > biggest_cluster: # TODO: Is this at all robust???
    #         ship_particle_label = j
    #         biggest_cluster = cluster_size

    ship_track_mask_small = IC_labels == ship_particle_label
    ship_track_mask = np.zeros(BTD.shape, dtype=bool)
    ship_track_mask[np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(var_aer)),np.logical_not(np.isnan(var_depth))),error_cluster_mask),size_mask)] = ship_track_mask_small

    # ship_track_mask = np.logical_and(np.logical_and(var_aer > 10, var_aer < 20), var_depth > 22)
    # ship_track_mask = np.logical_and(np.logical_and(var_aer > mean_size-5, var_aer < mean_size+5), var_depth > mean_depth-5)
    ship_track_mask = refl_var_ch06 > 0.4
    ###########################################################################


    ####TEMP######################
    # def phil_eq(depth):
    #     return 7 + depth**(3/10)

    def phil_eq(depth):
        # return 12 + depth**(2/90)
        return 7 + depth**(5/90)

    # var_depth = np.where(golden_arch_mask, np.nan, var_depth)
    # var_aer = np.where(golden_arch_mask, np.nan, var_aer)
    size_func = phil_eq(var_depth)
    # size_mask = np.logical_and(np.logical_and(var_aer > size_func-1, var_aer < size_func + 1), np.logical_and(var_aer < 20, var_depth < 60))
    # size_mask = np.logical_and(var_aer > size_func, var_aer < size_func+3)
    size_mask = np.logical_and(var_aer > size_func-3, var_aer < size_func+3)
    # var_aer = var_aer[size_mask]
    # var_depth = var_depth[size_mask]

    # filename = "early_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # labels_slice = np.zeros([BTD.shape[0], BTD.shape[1]])
    # labels_slice = np.where(golden_arch_mask, 255.0, labels_slice)
    # labels = np.zeros([BTD.shape[0], BTD.shape[1], 3], dtype=np.float32)
    # labels[:,:,2] = labels_slice
    # BTD_img = cv2.addWeighted(BTD_img, 1.0, labels, 0.5, 0)
    # cv2.imwrite(file_path, BTD_img)
    filename = "canny_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(file_path, img)

    filename = "NEW_MASK_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    labels_slice = np.zeros([BTD.shape[0], BTD.shape[1]])
    labels_slice = np.where(ship_track_mask, 255.0, labels_slice)
    # labels_slice = np.where(size_mask, 255.0, labels_slice)
    labels = np.zeros([BTD.shape[0], BTD.shape[1], 3], dtype=np.float32)
    labels[:,:,2] = labels_slice
    BTD_img = cv2.addWeighted(BTD_img, 1.0, labels, 0.5, 0)
    cv2.imwrite(file_path, BTD_img)

    # filename = "NEW_SCATTER_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt(var_aer[np.logical_not(np.isnan(ship_track_mask))], var_depth[np.logical_not(np.isnan(ship_track_mask))], ship_track_mask[np.logical_not(np.isnan(ship_track_mask))].astype(int), "km", fig2, ax2, file_path)

    # filename = "NEW_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin(var_aer, var_depth, fig2, ax2, file_path)

    # size_mask = var_aer < 20
    # var_aer = var_aer[size_mask]
    # var_depth = var_depth[size_mask]
    # depth_mask = var_depth < 60
    # var_aer = var_aer[depth_mask]
    # var_depth = var_depth[depth_mask]

    filename = "NEW_SCATTER_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    plot.scatter_plt_no_cluster_log(var_depth, var_aer, fig2, ax2, file_path)

    filename = "NEW_HEX_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    plot.hexbin_log(var_depth, var_aer, fig2, ax2, file_path)

    filename = "NEW_CLUSTER_" + str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    plot.scatter_plt_log(BTD_small, var_aer_small, IC_labels, "km", fig2, ax2, file_path)

    # filename = "ch2_ch6_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin(refl_var_ch02, refl_var_ch06, fig2, ax2, file_path)

    # filename = "ch2_ch6_scatter_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt_no_cluster(refl_var_ch02, refl_var_ch06, fig2, ax2, file_path)

    # filename = "size_ch2_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin(var_aer, refl_var_ch02, fig2, ax2, file_path)

    # filename = "size_ch2_scatter_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt_no_cluster(var_aer, refl_var_ch02, fig2, ax2, file_path)

    # filename = "size_ch6_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin(var_aer, refl_var_ch06, fig2, ax2, file_path)

    # filename = "size_ch6_scatter_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt_no_cluster(var_aer, refl_var_ch06, fig2, ax2, file_path)

    # filename = "size_BTD_HEX_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.hexbin(var_aer, BTD, fig2, ax2, file_path)

    # filename = "size_BTD_scatter_" + str(i) + ".png"
    # file_path = os.path.join(OUT_DIR, filename)
    # plot.scatter_plt_no_cluster(var_aer, BTD, fig2, ax2, file_path)

    norm_refls = (refl_var_ch02 - refl_var_ch06)/(refl_var_ch02 + refl_var_ch06)
    min_aer = np.nanmin(norm_refls)
    if min_aer < 0:
        norm_refls = norm_refls + np.abs(min_aer)
    max_aer = np.nanmax(norm_refls)
    norm_refls = norm_refls/max_aer
    norm_refls = cv2.cvtColor(norm_refls*255, cv2.COLOR_GRAY2BGR)
    filename_aer = "NEW_REFL_" + str(i) + ".png"
    file_path_aer = os.path.join(OUT_DIR, filename_aer)
    cv2.imwrite(file_path_aer, norm_refls)

    # min_aer = np.nanmin(var_aer)
    # if min_aer < 0:
    #     var_aer = var_aer + np.abs(min_aer)
    # max_aer = np.nanmax(var_aer)
    # var_aer = var_aer/max_aer
    # var_aer = cv2.cvtColor(var_aer*255, cv2.COLOR_GRAY2BGR)
    # filename_aer = "cloud_size_" + str(i) + ".png"
    # file_path_aer = os.path.join(OUT_DIR, filename_aer)
    # cv2.imwrite(file_path_aer, var_aer)

    # min_aer = np.nanmin(var_depth)
    # if min_aer < 0:
    #     var_depth = var_depth + np.abs(min_aer)
    # max_aer = np.nanmax(var_depth)
    # var_depth = var_depth/max_aer
    # var_depth = cv2.cvtColor(var_depth*255, cv2.COLOR_GRAY2BGR)
    # filename_aer = "cloud_depth_" + str(i) + ".png"
    # file_path_aer = os.path.join(OUT_DIR, filename_aer)
    # cv2.imwrite(file_path_aer, var_depth)
    ###############################
    
    image_list.append(BTD_img)
    BTD_list.append(BTD)
    refl_ch2_list.append(refl_var_ch02)
    refl_ch6_list.append(refl_var_ch06)
    golden_arch_list.append(golden_arch_mask) # NEXT 2 ARE TEMP
    high_cloud_list.append(high_cloud_mask)
    aer_list.append(var_aer)

    print("Image " + str(i) + " Calculated")
    i = i + 1


# TODO: Remove BTD_list in all areas if I am not using it for real final pngs
for i in range(len(BTD_list)):
    filename = str(i) + ".png"
    file_path = os.path.join(OUT_DIR, filename)
    boxes = trackers.get_boxes(i)

    BTD_img = image_list[i]
    BTD = BTD_list[i]
    refl_var_ch02 = refl_ch2_list[i]
    refl_var_ch06 = refl_ch6_list[i]
    var_aer = aer_list[i]

    # Make box plots for trackers
    # Also make and highlight the labels
    labels = np.zeros([BTD.shape[0], BTD.shape[1], 3], dtype=np.float32)
    j = 0 #TEMP######
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]

        if w > 0 and h > 0 and x >= 0 and y >= 0 and y+h <= BTD.shape[0] and x+w <= BTD.shape[1] and y < BTD.shape[0] and x < BTD.shape[1]:
            box_slice = BTD[y:y+h, x:x+w]
            ch2_slice = refl_var_ch02[y:y+h, x:x+w]
            ch6_slice = refl_var_ch06[y:y+h, x:x+w]
            aer_slice = var_aer[y:y+h, x:x+w]
            # ch2_and_ch6 = np.vstack((ch2_slice.flatten(), ch6_slice.flatten())).T

            ## inductive #########################################################################
            # clusterer = DBSCAN(eps=1.5, min_samples=25)
            # classifier = DecisionTreeClassifier()
            # inductive_cluster = InductiveClusterer(clusterer, classifier).fit(ch2_and_ch6)
            # kmeans_labels = inductive_cluster.predict(ch2_and_ch6) + 1

            # all_labels = np.unique(kmeans_labels)
            # max_refl = 0
            # track_label = 0
            # for k in all_labels:
            #     labeled_refl_array = ch6_slice.flatten()[kmeans_labels==k]
            #     mean_refl = np.nanmean(labeled_refl_array)
            #     if mean_refl > max_refl:
            #         track_label = k
            #         max_refl = mean_refl
            ######################################################################################

            ## Kmeans ####################################
            # km = KMeans(n_clusters=2)
            # kmeans_labels = km.fit_predict(ch2_and_ch6)
            # cluster_means = km.cluster_centers_[:, 1]
            #############################################

            labels_slice = labels[y:y+h, x:x+w, 2]
            labels_slice = np.where(np.logical_and(ch6_slice >= 0.28, ch2_slice >= 0.3), 255.0, labels_slice)
            # labels_slice = np.where(kmeans_labels.reshape(labels_slice.shape) == np.nanargmax(cluster_means), 255.0, labels_slice) #kmeans
            # labels_slice = np.where(kmeans_labels.reshape(labels_slice.shape) == track_label, 255.0, labels_slice) #inductive
            labels[y:y+h, x:x+w, 2] = labels_slice # Add red for labels

            ###### TESTING ###################
            filename2 = "BTD_box_" + str(j) + ".png"
            file_path2 = os.path.join(OUT_DIR, filename2)
            plot.scatter_plt_no_cluster(aer_slice,box_slice,fig2,ax2,file_path2)
            filename2 = "ch2_box_" + str(j) + ".png"
            file_path2 = os.path.join(OUT_DIR, filename2)
            plot.scatter_plt_no_cluster(aer_slice,ch2_slice,fig2,ax2,file_path2)
            filename2 = "ch6_box_" + str(j) + ".png"
            file_path2 = os.path.join(OUT_DIR, filename2)
            plot.scatter_plt_no_cluster(aer_slice,ch6_slice,fig2,ax2,file_path2)
            j = j + 1
            ##################################

            cv2.rectangle(BTD_img, (x, y), (x + w, y + h), (255.0, 0, 0), 2)

    ####TESTING######
    # labels_slice = np.zeros([BTD.shape[0], BTD.shape[1]])
    # labels_slice = np.where(golden_arch_list[i], 255.0, labels_slice)
    # # labels_slice = np.where(high_cloud_list[i], 255.0, labels_slice)
    # labels[:,:,2] = labels_slice
    ################

    BTD_img = cv2.addWeighted(BTD_img, 1.0, labels, 0.5, 0)

    cv2.imwrite(file_path, BTD_img)

    print("Image " + str(i) + " Complete")