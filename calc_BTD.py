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

# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/"
# DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-sep12/"
# DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-sep12/"
DATA_DIR_7 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-aug08/"
DATA_DIR_14 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-aug08/"
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
VMIN = -5
VMAX = 25
CMAP = "Greys"

############DATA###############

GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['07'],
              DateTimeIni = '20190807-220000', DateTimeFin = '20190809-020000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR_7)
GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['14'],
              DateTimeIni = '20190807-220000', DateTimeFin = '20190809-020000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR_14)

###########PROGRAM################

def bt_ch07_temp_conv(var):
    fk1 = 2.00774e05
    fk2 = 3.68909e03
    bc1 = 0.50777
    bc2 = 0.99929

    return (fk2/(np.log((fk1/var)+1))-bc1)/bc2 - 273.15

def bt_ch14_temp_conv(var):
    fk1 = 8.48310e03
    fk2 = 1.28490e03
    bc1 = 0.25361
    bc2 = 0.9991

    return (fk2/(np.log((fk1/var)+1))-bc1)/bc2 - 273.15

# Get contents of data dir for ch 7
data_list_7 = os.listdir(DATA_DIR_7)
if ".DS_Store" in data_list_7:
    data_list_7.remove(".DS_Store") # For mac users
data_list_7 = sorted(data_list_7)
first_ds_name = data_list_7.pop(0)
first_ds_path = os.path.join(DATA_DIR_7, first_ds_name)
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
var_ch07 = bt_ch07_temp_conv(var_ch07)

# if you want to use pcolormesh to plot data you will need calcute the corners of each pixel
loncor, latcor = GOES.get_lonlat_corners(lons, lats)

# Get contents of data dir for ch14
data_list_14 = os.listdir(DATA_DIR_14)
if ".DS_Store" in data_list_14:
    data_list_14.remove(".DS_Store") # For mac users
data_list_14 = sorted(data_list_14)
first_ds_name = data_list_14.pop(0)
first_ds_path = os.path.join(DATA_DIR_14, first_ds_name)
first_ds = Dataset(first_ds_path)
X = first_ds.variables['x']
Y = first_ds.variables['y']
var_ch14 = first_ds.variables["Rad"]
var_ch14, lons, lats, extra = GOES.slice_sat_image(var_ch14, X, Y, SatLon, SatHeight, SatSweep,
                                    LLLon, URLon, LLLat, URLat)
var_ch14 = np.where(lons==-999.99, np.nan, var_ch14)
var_ch14 = bt_ch14_temp_conv(var_ch14)

# Make BTD
var = var_ch07 - var_ch14

# Defines map projection
MapProj = ccrs.PlateCarree()

# Defines field projection
FieldProj = ccrs.PlateCarree()

# Creates figure
fig = plt.figure(dpi=150, figsize=(12, 9))
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=MapProj)
ax.set_extent(extents=[LLLon, URLon, LLLat, URLat], crs=MapProj)

# Set axes labels:
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Add geographic boundaries
countries = cf.NaturalEarthFeature(category='cultural', name='admin_0_countries',
                                   scale='50m', facecolor='none')
ax.add_feature(countries, edgecolor='black', linewidth=0.25)
states = cf.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                scale='50m', facecolor='none')
ax.add_feature(states, edgecolor='black', linewidth=0.25)

# Plot image
img = ax.pcolormesh(loncor, latcor, var, cmap = CMAP, transform = FieldProj)#, vmin=VMIN, vmax=VMAX)

# Customizing the plot border
ax.outline_patch.set_linewidth(0.3)

# Plot colorbar
cbar = plt.colorbar(img, extend='neither', spacing='proportional',
                    orientation = 'horizontal')
cbar.ax.tick_params(labelsize=6, labelcolor='black', width=0.5, direction='out', pad=1.0)
cbar.set_label(label='Brightness Temperature Difference (BTD)', size=6, color='black', weight='normal')
cbar.outline.set_linewidth(0.5)

# Sets X axis characteristics
xticks = np.arange(LLLon,URLon,2)
ax.set_xticks(xticks, crs=MapProj)
ax.set_xticklabels(xticks, fontsize=5.5, color='black')
lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°',
                                   dateline_direction_label=True)
ax.xaxis.set_major_formatter(lon_formatter)

# Sets Y axis characteristics
yticks = np.arange(LLLat,URLat,1)
ax.set_yticks(yticks, crs=MapProj)
ax.set_yticklabels(yticks, fontsize=5.5, color='black')
lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
ax.yaxis.set_major_formatter(lat_formatter)

# Sets grid characteristics
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, length=0.0, width=0.05)
# ax.gridlines(xlocs=xticks, ylocs=yticks, color='gray', alpha=0.6, draw_labels=False,
#              linewidth=0.25, linestyle='--')

ax.set_xlim(LLLon, URLon)
ax.set_ylim(LLLat, URLat)

# plt.savefig(os.path.join(OUT_DIR, '0.png'))
canvas = FigureCanvas(fig)
canvas.print_figure(os.path.join(OUT_DIR, '0.png'))

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

    var_ch07 = bt_ch07_temp_conv(var_ch07)
    var_ch14 = bt_ch14_temp_conv(var_ch14)

    # Make BTD
    var = var_ch07 - var_ch14

    img = ax.pcolormesh(loncor, latcor, var, cmap = CMAP, transform = FieldProj)#, vmin=VMIN, vmax=VMAX)
    # cbar = plt.colorbar(img, extend='neither', spacing='proportional',
    #                     orientation = 'horizontal')
    # cbar.ax.tick_params(labelsize=6, labelcolor='black', width=0.5, direction='out', pad=1.0)
    # cbar.set_label(label='Brightness Temperature Difference (BTD)', size=6, color='black', weight='normal')
    # cbar.outline.set_linewidth(0.5)
    
    canvas = FigureCanvas(fig)
    canvas.print_figure(file_path)

    print("Image " + str(i) + " Complete")
    i = i + 1