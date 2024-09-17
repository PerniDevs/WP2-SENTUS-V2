import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import conda
CondaFileDir = conda.__file__
CondaDir = CondaFileDir.split('lib')[0]
ProjLib = os.path.join(os.path.join(CondaDir, 'share'), 'proj')
os.environ["PROJ_LIB"] = ProjLib
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker
from GnssConstants import S_IN_H


import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# from COMMON.PlotsConstants import COLORS, MARKERS

# Adjust chunk size
plt.rcParams['agg.path.chunksize'] = 10000

def createFigure(PlotConf):
    try:
        if PlotConf["Type"] == "Lines":
            fig, ax = plt.subplots(1, 1, figsize = PlotConf["FigSize"])
        elif PlotConf["Type"] == "Polar":
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, figsize= PlotConf["FigSize"])
    
    except:
        fig, ax = plt.subplots(1, 1)

    return fig, ax

def saveFigure(fig, Path):
    Dir = os.path.dirname(Path)
    try:
        os.makedirs(Dir)
    except: pass
    fig.savefig(Path, dpi=150., bbox_inches='tight')

def prepareAxis(PlotConf, ax):
    for key in PlotConf:
        if key == "Title":
            ax.set_title(PlotConf["Title"])

        for axis in ["x", "y"]:
            if axis == "x":
                if key == axis + "Label":
                    ax.set_xlabel(PlotConf[axis + "Label"])

                if key == axis + "Ticks":
                    ax.set_xticks(PlotConf[axis + "Ticks"])

                if key == axis + "TicksLabels":
                    ax.set_xticklabels(PlotConf[axis + "TicksLabels"])
                
                if key == axis + "Lim":
                    ax.set_xlim(PlotConf[axis + "Lim"])

            if axis == "y":
                if key == axis + "Label":
                    ax.set_ylabel(PlotConf[axis + "Label"])

                if key == axis + "Ticks":
                    ax.set_yticks(PlotConf[axis + "Ticks"])

                if key == axis + "TicksLabels":
                    ax.set_yticklabels(PlotConf[axis + "TicksLabels"])
                
                if key == axis + "Lim":
                    ax.set_ylim(PlotConf[axis + "Lim"])

        if key == "Grid" and PlotConf[key] == True:
            ax.grid(linestyle='--', linewidth=0.5, which='both')

def preparePolarAxis(PlotConf, ax):
    for key in PlotConf:
        if key == "Title":
            ax.set_title(PlotConf["Title"])

        for axis in ["theta", "r"]:
            if axis == "theta":

                if key == axis + "TicksLabels":
                    ax.set_xticklabels(PlotConf[axis + "TicksLabels"])
                
                if key == axis + "ZeroLocation":
                    ax.set_theta_zero_location(PlotConf[axis + "ZeroLocation"])

            if axis == "r":
    
                if key == axis + "LabelPos":
                    ax.set_rlabel_position(PlotConf[axis + "LabelPos"])

                if key == axis + "Ticks":
                    ax.set_rticks(PlotConf[axis + "Ticks"])
                
                if key == axis + "Lim":
                    ax.set_rlim(PlotConf[axis + "Lim"])

        if key == "Grid" and PlotConf[key] == True:
            ax.grid(linestyle='--', linewidth=0.5, which='both')

def prepareColorBar(PlotConf, ax, Values, scatter = None):
    try:
        Min = PlotConf["ColorBarMin"]
    except:
        Mins = []
        for v in Values.values():
            Mins.append(min(v))
        Min = min(Mins)
    try:
        Max = PlotConf["ColorBarMax"]
    except:
        Maxs = []
        for v in Values.values():
            Maxs.append(max(v))
        Max = max(Maxs)

    divider = make_axes_locatable(ax) 
    color_ax = divider.append_axes("right", size="3%", pad="2%")
    cmap = mpl.cm.get_cmap(PlotConf["ColorBar"])

    if "ColorBarBins" in PlotConf: 
        num_bins = PlotConf["ColorBarBins"]
        bounds = np.linspace(Min, Max, num_bins + 1)
        normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbar = mpl.colorbar.ColorbarBase(color_ax, 
                                         cmap=cmap, 
                                         norm=normalize,
                                         label=PlotConf["ColorBarLabel"],
                                         boundaries=bounds,
                                         ticks=bounds)
    else:
        # Handle the case where ColorBarBins is not provided or other KeyError
        normalize = mpl.cm.colors.Normalize(vmin=Min, vmax=Max)
    
        # Create a continuous colorbar
        cbar = mpl.colorbar.ColorbarBase(color_ax, cmap=cmap, norm=normalize,
                                     label=PlotConf["ColorBarLabel"])
    
    try:
        # if "ColorBarBins" in PlotConf:
        #     # Adjust tick labels to display the bin values
        #     tick_labels = [f'{tick:f}' for tick in tick_positions]
        #     cbar.set_ticklabels(tick_labels)
        # else:
        cbar.set_ticks(PlotConf["ColorBarSetTicks"])
        cbar.set_ticklabels(PlotConf["ColorBarSetTicks"])
    except:
        print(" Colobar Ticks not applicable\n")

    return normalize, cmap

def preparePolarColorbar(PlotConf, ax, Values):
    try:
        Min = PlotConf["ColorBarMin"]
    except:
        Mins = []
        for v in Values.values():
            Mins.append(min(v))
        Min = min(Mins)
    try:
        Max = PlotConf["ColorBarMax"]
    except:
        Maxs = []
        for v in Values.values():
            Maxs.append(max(v))
        Max = max(Maxs)
    
    normalize = plt.Normalize(vmin=Min, vmax=Max)
    cmap = PlotConf["ColorBar"]

    # scatter plot with PRN mapped to color
    for Label in PlotConf["rData"].keys():
        scatter = ax.scatter(PlotConf["thetaData"][Label], PlotConf["rData"][Label], c=PlotConf["zData"][Label], cmap=cmap, linewidths=1, marker="|", s=1)

    # Create the colorbar with PRN label
    cbar = plt.colorbar(scatter, ax=ax, norm=normalize)
    cbar.set_label('GPS-PRN')
    cbar.set_ticks(PlotConf["ColorBarSetTicks"])
    cbar.set_ticklabels(PlotConf["ColorBarSetTicks"])

    return ax

def drawMap(PlotConf, ax,):
    Map = Basemap(projection = 'cyl',
    llcrnrlat  = PlotConf["LatMin"]-0,
    urcrnrlat  = PlotConf["LatMax"]+0,
    llcrnrlon  = PlotConf["LonMin"]-0,
    urcrnrlon  = PlotConf["LonMax"]+0,
    lat_ts     = 10,
    resolution = 'l',
    ax         = ax)

    # Draw map meridians
    Map.drawmeridians(
    np.arange(PlotConf["LonMin"],PlotConf["LonMax"]+1,PlotConf["LonStep"]),
    labels = [0,0,0,1],
    fontsize = 6,
    linewidth=0.2)
        
    # Draw map parallels
    Map.drawparallels(
    np.arange(PlotConf["LatMin"],PlotConf["LatMax"]+1,PlotConf["LatStep"]),
    labels = [1,0,0,0],
    fontsize = 6,
    linewidth=0.2)

    # Draw coastlines
    Map.drawcoastlines(linewidth=0.5)

    # Draw countries
    Map.drawcountries(linewidth=0.25)

def generateLinesPlot(PlotConf):
    LineWidth = 1.5

    fig, ax = createFigure(PlotConf)

    prepareAxis(PlotConf, ax)

    for key in PlotConf:
        if key == "LineWidth":
            LineWidth = PlotConf["LineWidth"]
        if key == "ColorBar":
            normalize, cmap = prepareColorBar(PlotConf, ax, PlotConf["zData"])
        if key == "Map" and PlotConf[key] == True:
            drawMap(PlotConf, ax)
    
    ax2 = None
    try:
        if PlotConf["MultiAxis"]:
            ax2 = ax.twinx()
            ax2.set_ylabel(PlotConf["yLabel2"])
            ax2.set_ylim(PlotConf["yLim2"])
            ax2.set_yticks(PlotConf["yTicks2"])
    except:
        print(" No multiaxes detected ... \n")

    for Label in PlotConf["yData"].keys():
        if "ColorBar" in PlotConf:
            colors = cmap(normalize(np.array(PlotConf["zData"][Label])))

            if "Flags" in PlotConf:
                flags = PlotConf["Flags"][Label]
                # Apply grey where flag is 1
                colors[flags != 1] = mpl.colors.to_rgba("gray")
            else:
                pass
                  
            try:
                ax.scatter(PlotConf["xData"][Label], PlotConf["yData"][Label], 
                marker = PlotConf["Marker"],
                linewidth = LineWidth,
                s = PlotConf['s'],
                c = colors, 
                zorder=1)
    
                if "Annotations" in PlotConf and Label in PlotConf["Annotations"]:
                    x_data = np.array(PlotConf["xData"][Label])
                    y_data = np.array(PlotConf["yData"][Label])
                    annotations = np.array(PlotConf["Annotations"][Label])
                    prev_x_data_point = 0
                    for i, text in enumerate(annotations):
                        text_color = colors[i][:3]

                        # Alternating Offsets
                        if i %2 ==0:
                            offset = 10
                        else:
                            offset = -5
                        
                        # if x_data[i] > prev_x_data_point +700 / S_IN_H:
                        ax.annotate(text, 
                                        (x_data[i], y_data[i]), 
                                        fontsize=8, 
                                        ha='center', 
                                        va="top",  
                                        color=text_color, 
                                        xytext=(0, offset), 
                                        textcoords='offset points',
                                        # bbox=dict(boxstyle='round,pad=0.3', edgecolor=(0, 0, 0, 0), facecolor='white')
                                        )
                            
                            # prev_x_data_point = x_data[i]
                        
                else:
                    pass

            except:
                ax.scatter(PlotConf["xData"][Label], PlotConf["yData"][Label], 
                marker = PlotConf["Marker"],
                linewidth = LineWidth,
                c = colors,
                zorder=1)

        else:
            if Label == 0 and ax2: 
                ax2.plot(PlotConf["xData"][Label], PlotConf["yData"][Label],
                         PlotConf["Marker"],
                         linewidth=LineWidth,
                         color=PlotConf["c"][Label],
                         label=PlotConf["Label"][Label],
                         linestyle=PlotConf["LineStyle"])

            else:
                try:
                    ax.plot(PlotConf["xData"][Label], PlotConf["yData"][Label],
                            PlotConf["Marker"],
                            linewidth=LineWidth,
                            color=PlotConf["c"][Label],
                            label=PlotConf["Label"][Label],
                            linestyle=PlotConf["LineStyle"])
                except:
                    try:
                        ax.plot(PlotConf["xData"][Label], PlotConf["yData"][Label],
                                PlotConf["Marker"],
                                linewidth=LineWidth,
                                color=PlotConf["c"][Label],
                                linestyle=PlotConf["LineStyle"])
                    except:
                        ax.plot(PlotConf["xData"][Label], PlotConf["yData"][Label],
                        PlotConf["Marker"],color=PlotConf["c"][Label],
                        linewidth = LineWidth)

            try:    
                # Combine legends from both axes
                handles1, labels1 = ax.get_legend_handles_labels()
                if ax2:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles = handles1 + handles2
                    labels = labels1 + labels2
                else:
                    handles = handles1
                    labels = labels1

                ax.legend(handles, labels, loc=PlotConf["LabelLoc"])
            except:
                print(" No labels to be applied ... \n")

    try:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(PlotConf["yDataFormatter"]))
    except:
        print(' No Formatter for y-axis \n')

    saveFigure(fig, PlotConf["Path"])

def generatePolarPlot(PlotConf):
    LineWidth = 1.5

    fig, ax = createFigure(PlotConf)

    preparePolarAxis(PlotConf, ax)

    for key in PlotConf:
        if key == "LineWidth":
            LineWidth = PlotConf["LineWidth"]
        if key == "ColorBar":
            ax = preparePolarColorbar(PlotConf, ax, PlotConf["zData"])
        if key == "Map" and PlotConf[key] == True:
            drawMap(PlotConf, ax)

    saveFigure(fig, PlotConf["Path"])

def generatePlot(PlotConf):
    if(PlotConf["Type"] == "Lines"):
        generateLinesPlot(PlotConf)
    elif(PlotConf["Type"] == "Polar"):
        generatePolarPlot(PlotConf)