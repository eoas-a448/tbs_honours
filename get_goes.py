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

# DATA_DIR = "/Users/tschmidt/repos/tgs_honours/data/"
DATA_DIR1 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-apr24/" # This one needs trailing / for goes lib
DATA_DIR2 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-apr24/" # This one needs trailing / for goes lib
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"
# Defines the plot area
LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
# Ch02 min/max
# VMIN=0
# VMAX=250
VMIN = -15
VMAX = 40
isIR = True

GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['07'],
              DateTimeIni = '20190424-140000', DateTimeFin = '20190425-010000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)
GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['14'],
              DateTimeIni = '20190424-140000', DateTimeFin = '20190425-010000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR2)