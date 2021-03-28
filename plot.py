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

def scatter_plt(var1, var2, labels, fig, ax, out_file):
    label_color = [CLUSTER_COLORS[l] for l in labels]
    ax.scatter(var1.flatten(), var2.flatten(), c=label_color, s=1)
    
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

def scatter_plt_no_cluster_log(var1, var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), s=0.01)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size")
    # ax.set_xticks(np.arange(min(var1),max(var1), 1))
    # ax.set_yticks(np.arange(min(var2),max(var2), 1))
    
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

def hexbin_log(var1, var2, fig, ax, out_file):
    ax.hexbin(var1.flatten(), var2.flatten(), xscale="log", yscale="log")
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size")
    # ax.set_xticks(np.arange(min(var1),max(var1), 1))
    # ax.set_yticks(np.arange(min(var2),max(var2), 1))

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

    plt.cla()

def scatter_plt_log(var1, var2, labels, fig, ax, out_file):
    label_color = [CLUSTER_COLORS[l] for l in labels]
    ax.scatter(var1.flatten(), var2.flatten(), c=label_color, s=0.01)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size")
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file)

    plt.cla()