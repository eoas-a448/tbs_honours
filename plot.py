import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

def scatter_plt_2_cluster_legend(var1, var2, cluster_var1, cluster_var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), c='b', s=0.01, label="Other Data")
    ax.scatter(cluster_var1.flatten(), cluster_var2.flatten(), c='r', s=0.01, label="Open Ocean")
    ax.set_xlabel("Brightness Temperature Difference (°C)")
    ax.set_ylabel("Channel 2 Radiances $(Wm^{-1}sr^{-1}um^{-1})$")
    ax.set_xlim([5,20])
    leg = ax.legend()
    leg.legendHandles[0]._sizes = [40]
    leg.legendHandles[1]._sizes = [40]
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def scatter_plt(var1, var2, labels, fig, ax, out_file):
    label_color = [CLUSTER_COLORS[l] for l in labels]
    ax.scatter(var1.flatten(), var2.flatten(), c=label_color, s=0.01)
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def scatter_plt_no_cluster(var1, var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), s=0.01)
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def scatter_plt_no_cluster_log(var1, var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), s=0.01)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size (μm)")
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def hexbin(var1, var2, i, fig, ax, out_file):
    hb = ax.hexbin(var1.flatten(), var2.flatten(), vmin=0, vmax=1500)
    ax.set_xlabel("Brightness Temperature Difference (°C)")
    ax.set_ylabel("Channel 2 Radiances $(Wm^{-1}sr^{-1}um^{-1})$")
    ax.set_xlim([5,20])
    if i == 0:
        cb = fig.colorbar(hb, ax=ax, extend="max")
        cb.set_label('Pixel Counts')

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def hexbin_log(var1, var2, i, fig, ax, out_file):
    hb = ax.hexbin(var1.flatten(), var2.flatten(), xscale="log", yscale="log", vmin=0, vmax=300)
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size (μm)")
    if i == 0:
        cb = fig.colorbar(hb, ax=ax, extend="max")
        cb.set_label('Pixel Counts')

    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()

def scatter_plt_log(var1, var2, cluster_var1, cluster_var2, fig, ax, out_file):
    ax.scatter(var1.flatten(), var2.flatten(), c='b', s=0.01, label="Other Data")
    ax.scatter(cluster_var1.flatten(), cluster_var2.flatten(), c='r', s=0.01, label="Possible Ship Tracks")
    ax.set_xlabel("Cloud Optical Depth")
    ax.set_ylabel("Cloud Particle Size (μm)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    leg = ax.legend()
    leg.legendHandles[0]._sizes = [40]
    leg.legendHandles[1]._sizes = [40]
    
    # Save image
    canvas = FigureCanvas(fig)
    canvas.print_figure(out_file, bbox_inches='tight')

    ax.clear()