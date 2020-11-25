import calc_BTD

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

CMAP = "Greys"
# VMIN = -20
# VMAX = 30
VMIN = 0 #These are for radiances
VMAX = 130

def main_func(var, loncor, latcor, fig, ax, MapProj, FieldProj, out_file):

    # Plot image
    ax.pcolormesh(loncor, latcor, var, cmap = CMAP, transform = FieldProj)#, vmin=VMIN, vmax=VMAX)

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

def scatter_plt(var1, var2, out_file):
    fig = plt.figure(dpi=150, figsize=(12, 9))
    ax = fig.add_axes([0.1, 0.16, 0.80, 0.75])
    ax.scatter(var1, var2)
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)