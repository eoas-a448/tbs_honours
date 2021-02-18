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
CLUSTER_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def main_func(var, loncor, latcor, fig, ax, MapProj, FieldProj, out_file):

    # Plot image
    ax.pcolormesh(loncor, latcor, var, cmap = CMAP, transform = FieldProj)#, vmin=VMIN, vmax=VMAX)

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

def scatter_plt(var1, var2, labels, km, fig, ax, out_file):
    label_color = [CLUSTER_COLORS[l] for l in labels]
    ax.scatter(var1.flatten(), var2.flatten(), c=label_color, s=1)
    # ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='*')
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

    plt.cla()

def scatter_plt_no_cluster(var1, var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), s=1)
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

    plt.cla()

def hexbin(var1, var2, fig, ax, out_file):
    ax.hexbin(var1.flatten(), var2.flatten())

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

    plt.cla()