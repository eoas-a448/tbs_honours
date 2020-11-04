import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
from matplotlib import colors

LLLon, URLon = -135, -116.5
LLLat, URLat = 28, 38.5
# VMIN = -5
# VMAX = 25
# VMIN = -20
# VMAX = 30
# VMIN = 0 #These two are for radiances
# VMAX = 130
CMAP = "Greys"

def main_func():
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

    # Customizing the plot border
    ax.outline_patch.set_linewidth(0.3)

    #Plot colorbar
    # norm = colors.Normalize(vmin=VMIN, vmax=VMAX)
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, extend='neither', spacing='proportional',
    #                     orientation = 'horizontal')
    # cbar.ax.tick_params(labelsize=6, labelcolor='black', width=0.5, direction='out', pad=1.0)
    # cbar.set_label(label='Brightness Temperature Difference (BTD)', size=6, color='black', weight='normal')
    # cbar.outline.set_linewidth(0.5)

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

    ax.set_xlim(LLLon, URLon)
    ax.set_ylim(LLLat, URLat)

    return fig, ax, MapProj, FieldProj