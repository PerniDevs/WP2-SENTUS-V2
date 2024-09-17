#!/usr/bin/env python

########################################################################
# PreprocessingPlots.py:
# This is the PreprocessingPlots Module of SENTUS tool
#
#  Project:        SENTUS
#  File:           PreprocessingPlots.py
#
#   Author: Agustin Pernigotti
#   Copyright 2024 Agustin Pernigotti
#
# -----------------------------------------------------------------
# Date       | Author             | Action
# -----------------------------------------------------------------
#
########################################################################

import sys, os
from pandas import unique
from pandas import read_csv
from InputOutput import PreproIdx
from InputOutput import REJECTION_CAUSE_DESC
sys.path.append(os.getcwd() + '/' + \
    os.path.dirname(sys.argv[0]) + '/' + 'COMMON')
from COMMON import GnssConstants
from COMMON.Plots import generatePlot
from COMMON.allPRNs import allprns


def initPlot(PreproObsFile, PlotConf, Title, Label):
    PreproObsFileName = os.path.basename(PreproObsFile)
    PreproObsFileNameSplit = PreproObsFileName.split('_')
    Rcvr = PreproObsFileNameSplit[2]
    DatepDat = PreproObsFileNameSplit[3]
    Date = DatepDat.split('.')[0]
    Year = Date[1:3]
    Doy = Date[4:]

    PlotConf["xLabel"] = "Hour of Day %s" % Doy

    PlotConf["Title"] = "%s from %s on Year %s"\
        " DoY %s" % (Title, Rcvr, Year, Doy)

    PlotConf["Path"] = sys.argv[1] + '/OUT/PPVE/' + \
        '%s_%s_Y%sD%s.png' % (Label, Rcvr, Year, Doy)
    
    return PlotConf


# Function to convert 'G01', 'G02', etc. to 1, 2, etc.
def convert_satlabel_to_prn(value):
    return int(value[1:])


# Function to convert 'G01', 'G02', etc. to 'G'
def convert_satlabel_to_const(value):
    return value[0]


# Plot Satellite Visibility
def plotSatVisibility(PreproObsFile, PreproObsData):

    all_prns = allprns()

    PlotConf = {}
    PlotConf["Type"] = "Lines"
    PlotConf["FigSize"] = (16.4,14.6)
    PlotConf["Title"] = "Satellite Visibility from s6an on Year 24 DoY 011"
       
    # PlotConf["yLabel"] = "GPS-GAL-PRN"
    # PlotConf["yTicks"] = range(1, len(all_prns.values()))
    # PlotConf["yTicksLabels"] = sorted(all_prns.keys())
    # PlotConf["yLim"] = [0, len(all_prns.values())]

    PlotConf["yLabel"] = "GPS-GAL-PRN"
    PlotConf["yTicks"] = range(0, len(sorted(unique(PreproObsData[PreproIdx["PRN"]]))))
    PlotConf["yTicksLabels"] = sorted(unique(PreproObsData[PreproIdx["PRN"]]))
    PlotConf["yLim"] = [-0.5, len(sorted(unique(PreproObsData[PreproIdx["PRN"]])))]

    PlotConf["xLabel"] = "Hour of DoY 011"
    PlotConf["xTicks"] = range(0, 25)
    PlotConf["xLim"] = [0, 24]

    PlotConf["Grid"] = 1

    PlotConf["Marker"] = '.'
    PlotConf["LineWidth"] = 1.5

    PlotConf["ColorBar"] = "gnuplot"
    PlotConf["ColorBarLabel"] = "Elevation [deg]"
    PlotConf["ColorBarMin"] = 0.
    PlotConf["ColorBarMax"] = 90.

    PlotConf["xData"] = {}
    PlotConf["yData"] = {}
    PlotConf["zData"] = {}
    PlotConf["Flags"] = {}

    # for prn_key, prn_value in all_prns.items():

    for prn in sorted(unique(PreproObsData[PreproIdx["PRN"]])):
        FilterCond = PreproObsData[PreproIdx["PRN"]] == prn
        PlotConf["xData"][prn] = PreproObsData[PreproIdx["SOD"]][FilterCond] / GnssConstants.S_IN_H
        PlotConf["yData"][prn] = PreproObsData[PreproIdx["PRN"]][FilterCond]
        PlotConf["zData"][prn] = PreproObsData[PreproIdx["ELEV"]][FilterCond]
        PlotConf["Flags"][prn] = PreproObsData[PreproIdx["STATUS"]][FilterCond]

    PlotConf["Path"] = sys.argv[1] + '/OUT/PPVE/SAT/' + 'SAT_VISIBILITY_s6an_D011Y24.png'

    # Debugging output
    generatePlot(PlotConf)


# Plot Number of Satellites
def plotNumSats(PreproObsFile, PreproObsData):

    # Divide data frame
    # Raw Data
    PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E")]
    PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G")]

    # Smoothed data
    PreproObsDataGalileoSmoothed = PreproObsDataGalileo[PreproObsData[PreproIdx["STATUS"]] == 1]
    PreproObsDataGPSSmoothed = PreproObsDataGPS[PreproObsDataGPS[PreproIdx["STATUS"]] == 1]
    PreproObsDataSmoothed = PreproObsData[PreproObsData[PreproIdx["STATUS"]] == 1]

    # Set Conf Dicts
    PlotConfGalileo = {
        "Type": "Lines",
        "FigSize" : (10.4, 6.6),

        "Title" : "Number of GAL Satellites from s6an on Year 24 DoY 011",
        "yLabel" : "Number of Satellites",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
        "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

        "yLim" : [0, 20],
        "yTicks" : range(0, 20),

        "Grid" : 1,
        "c" : {0: "orange", 1: "green"},
        "Marker" : "",
        "LineWidth" : 1,
        "LineStyle" : "-",

        "Label" : {0: "RAW", 1: "SMOOTHED"},
        "LabelLoc" : "upper left",
        
        "xData": {
            0: unique(PreproObsDataGalileo[PreproIdx["SOD"]]) / GnssConstants.S_IN_H,
            1: unique(PreproObsDataGalileoSmoothed[PreproIdx["SOD"]]) / GnssConstants.S_IN_H
        },

        "yData": {
            0: PreproObsDataGalileo.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count(),
            1: PreproObsDataGalileoSmoothed.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count()
        },

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'NUMBER_OF_GAL_SATELLITES_s6an_D011Y24.png',
    }

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (10.4, 6.6),

        "Title" : "Number of GPS Satellites from s6an on Year 24 DoY 011",
        "yLabel" : "Number of Satellites",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
        "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

        "yLim" : [0, 20],
        "yTicks" : range(0, 20),

        "Grid" : 1,
        "c" : {0: "orange", 1: "green"},
        "Marker" : "",
        "LineWidth" : 1,
        "LineStyle" : "-",

        "Label" : {0: "RAW", 1: "SMOOTHED"},
        "LabelLoc" : "upper left",
        
        "xData": {
            0: unique(PreproObsDataGPS[PreproIdx["SOD"]]) / GnssConstants.S_IN_H,
            1: unique(PreproObsDataGPSSmoothed[PreproIdx["SOD"]]) / GnssConstants.S_IN_H
        },

        "yData": {
            0: PreproObsDataGPS.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count(),
            1: PreproObsDataGPSSmoothed.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count()
        },

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'NUMBER_OF_GPS_SATELLITES_s6an_D011Y24.png',
    }

    PlotConf = {
        "Type": "Lines",
        "FigSize" : (10.4, 6.6),

        "Title" : "Number of GPS+GAL from s6an on Year 24 DoY 011",
        "yLabel" : "Number of Satellites",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(round(PreproObsData[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsData[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
        "xLim" : [round(PreproObsData[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsData[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

        "yLim" : [0, 20],
        "yTicks" : range(0, 20),

        "Grid" : 1,
        "c" : {0: "orange", 1: "green"},
        "Marker" : "",
        "LineWidth" : 1,
        "LineStyle" : "-",

        "Label" : {0: "RAW", 1: "SMOOTHED"},
        "LabelLoc" : "upper left",

        "xData": {
            0: unique(PreproObsData[PreproIdx["SOD"]]) / GnssConstants.S_IN_H,
            1: unique(PreproObsDataSmoothed[PreproIdx["SOD"]]) / GnssConstants.S_IN_H
        },

        "yData": {
            0: PreproObsData.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count(),
            1: PreproObsDataSmoothed.groupby(PreproIdx["SOD"])[PreproIdx["STATUS"]].count()
        },

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'NUMBER_OF_GPS+GAL_SATELLITES_s6an_D011Y24.png',

    }

    data_array = [PlotConfGalileo, PlotConfGPS, PlotConf]

    for conf in data_array:
        generatePlot(conf)


# Plot Code IF - Code IF Smoothed
def plotIFIFSmoothed(PreproObsFile, PreproObsData):

    PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E")]
    PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G")]

    # GALILEO data
    min_val_gal = int(min(PreproObsDataGalileo[PreproIdx["CODE_IF"]] - PreproObsDataGalileo[PreproIdx["SMOOTH_IF"]]))
    max_val_gal = int(max(PreproObsDataGalileo[PreproIdx["CODE_IF"]] - PreproObsDataGalileo[PreproIdx["SMOOTH_IF"]]))

    # GPS data
    min_val_gps = int(min(PreproObsDataGPS[PreproIdx["CODE_IF"]] - PreproObsDataGPS[PreproIdx["SMOOTH_IF"]]))
    max_val_gps = int(max(PreproObsDataGPS[PreproIdx["CODE_IF"]] - PreproObsDataGPS[PreproIdx["SMOOTH_IF"]]))

    PlotConfGalileo = {
        "Type": "Lines",
        "FigSize" : (8.4, 6.6),

        "Title" : "GAL Code IF - Code IF Smoothed from s6an on Year 24 DoY 011",
        "yLabel" : "GAL Code IF - Code IF Smoothed [m]",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
        "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

        "yLim" : [-2, 2],
        "yTicks" : range(-2, 3),

        "Grid" : 1,
        
        "Marker" : ".",
        "LineWidth" : 0,

        "ColorBar": "gnuplot",
        "ColorBarLabel": "Elevation [deg]",
        "ColorBarMin" : 0,
        "ColorBarMax" : 90,
        
        "s" : 20,
        
        "Label" : {0},
        
        "xData": {
            0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
        },

        "yData": {
            0: PreproObsDataGalileo[PreproIdx["CODE_IF"]] - PreproObsDataGalileo[PreproIdx["SMOOTH_IF"]],  
        },

        "zData":{
            0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
        },

        "Flags":{
            0: PreproObsDataGalileo[PreproIdx["STATUS"]],
        },

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'GAL_CODEIF_SMOOTHEDIF_s6an_D011Y24.png',
    }

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (8.4, 6.6),

        "Title" : "GPS Code IF - Code IF Smoothed from s6an on Year 24 DoY 011",
        "yLabel" : "GPS Code IF - Code IF Smoothed [m]",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
        "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

        "yLim": [-4 , 4],
        "yTicks" : range(-4, 5),

        "Grid" : 1,
        
        "Marker" : ".",
        "LineWidth" : 0,

        "ColorBar": "gnuplot",
        "ColorBarLabel": "Elevation [deg]",
        "ColorBarMin" : 0,
        "ColorBarMax" : 90,
        
        "s" : 20,
        
        "Label" : {0},
        
        "xData": {
            0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
        },

        "yData": {
            0: PreproObsDataGPS[PreproIdx["CODE_IF"]] - PreproObsDataGPS[PreproIdx["SMOOTH_IF"]], 
        },

        "zData":{
            0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
        },

        "Flags":{
            0: PreproObsDataGPS[PreproIdx["STATUS"]],
        },

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'GPS_CODEIF_SMOOTHEDIF_s6an_D011Y24.png',
    }

    all_confs = [PlotConfGalileo, PlotConfGPS]

    for conf in all_confs:
        generatePlot(conf)


# Plot C/N0
def plotCN0(PreproObsFile, PreproObsData, PlotTitle, PlotLabel):

    if PlotTitle == "CN0_F1" or  PlotLabel == "S1":
        
        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["S1"]]]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["S1"]]]

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL CN0_F1 [dB-Hz]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [PreproObsDataGalileo[PreproIdx["S1"]].min(), PreproObsDataGalileo[PreproIdx["S1"]].max() + 1],
            "yTicks" : range(int(PreproObsDataGalileo[PreproIdx["S1"]].min()) -5, int(PreproObsDataGalileo[PreproIdx["S1"]].max()) +5, 5),

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["S1"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)
        
        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS CN0_F1 [dB-Hz]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [PreproObsDataGPS[PreproIdx["S1"]].min(), PreproObsDataGPS[PreproIdx["S1"]].max() + 1],
            "yTicks" : range(int(PreproObsDataGPS[PreproIdx["S1"]].min()) -5, int(PreproObsDataGPS[PreproIdx["S1"]].max()) +5, 5 ),

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["S1"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGPS]

        for conf in all_confs:
            generatePlot(conf)

    elif PlotTitle == "CN0_F2" or  PlotLabel == "S2":

        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["S2"]]]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["S2"]]]

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL CN0_F2 [dB-Hz]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [PreproObsDataGalileo[PreproIdx["S2"]].min(), PreproObsDataGalileo[PreproIdx["S2"]].max() + 1],
            "yTicks" : range(int(PreproObsDataGalileo[PreproIdx["S2"]].min()) -5, int(PreproObsDataGalileo[PreproIdx["S2"]].max()) + 5, 5 ),

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["S2"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)

        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS CN0_F2 [dB-Hz]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [PreproObsDataGPS[PreproIdx["S2"]].min(), PreproObsDataGPS[PreproIdx["S2"]].max() + 1],
            "yTicks" : range(int(PreproObsDataGPS[PreproIdx["S2"]].min()) -5, int(PreproObsDataGPS[PreproIdx["S2"]].max()) +5, 5),

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["S2"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGPS]

        for conf in all_confs:
            generatePlot(conf)


# Plot Rejection Flags
def plotRejectionFlags(PreproObsFile, PreproObsData):
    all_prns = allprns()

    gal_prn = [int(convert_satlabel_to_prn(const)) for const, _ in all_prns.items() if const.startswith("E")]
    gps_prn = [int(convert_satlabel_to_prn(const)) for const, _ in all_prns.items() if const.startswith("G")]

    PreproObsDataGalileo = PreproObsData[(PreproObsData[PreproIdx["PRN"]].str.startswith("E") & (PreproObsData[PreproIdx["REJECT"]] != 0))]
    PreproObsDataGPS =     PreproObsData[(PreproObsData[PreproIdx["PRN"]].str.startswith("G") & (PreproObsData[PreproIdx["REJECT"]] != 0))]

    # Aggregate data by 1000s intervals
    interval = 1000  # seconds
    PreproObsDataGalileo['Interval'] = (PreproObsDataGalileo[PreproIdx["SOD"]] // interval) * interval
    aggregated_dataGalileo = PreproObsDataGalileo.groupby('Interval').apply(lambda x: x.drop_duplicates(subset=[PreproIdx["REJECT"]]))

    PreproObsDataGPS['Interval'] = (PreproObsDataGPS[PreproIdx["SOD"]] // interval) * interval
    aggregated_dataGPS = PreproObsDataGPS.groupby('Interval').apply(lambda x: x.drop_duplicates(subset=[PreproIdx["REJECT"]]))

    PlotConfGalileo = {
        "Type": "Lines",
        "FigSize" : (10.4, 6.6),

        "Title" : "GAL Rejection Flags from s6an on Year 24 DoY 011",
        "yLabel" : "GAL Rejection Flags",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(int(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H)), int(round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)) + 1),
        "xLim" : [int(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H)), int(round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H))],

        "yLim" : [0, len(REJECTION_CAUSE_DESC.keys()) + 1],
        "yTicks" : range(1, len(REJECTION_CAUSE_DESC.keys()) + 1 ),
        "yTicksLabels" : REJECTION_CAUSE_DESC.keys(), 

        "Marker" : ".",
        "LineWidth" : 0,
        "Grid" : 1,

        "ColorBar" : "nipy_spectral",
        "ColorBarLabel" : "Galileo PRN",
        "ColorBarMin" : min(gal_prn),
        "ColorBarMax" : max(gal_prn),
        "ColorBarSetTicks": sorted(gal_prn),
        "ColorBarBins": len(gal_prn), 

        "s" : 20, 
        "Label" : 0,
        
        "Annotations": {0: aggregated_dataGalileo[PreproIdx["PRN"]]},

        "xData": {0: aggregated_dataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H},
        "yData": {0: aggregated_dataGalileo[PreproIdx["REJECT"]]},
        "zData" : {0: [int(convert_satlabel_to_prn(prn)) for prn in aggregated_dataGalileo[PreproIdx["PRN"]]]}, 

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'GAL_REJECTION_FLAGS_s6an_D011Y24.png',
    }

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (10.4, 6.6),

        "Title" : "GPS Rejection Flags from s6an on Year 24 DoY 011",
        "yLabel" : "GPS Rejection Flags",
        "xLabel" : "Hour of DoY 011",

        "xTicks": range(int(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H)), int(round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)) + 1),
        "xLim" : [int(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H)), int(round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H))],

        "yLim" : [0, len(REJECTION_CAUSE_DESC.keys()) + 1],
        "yTicks" : range(1, len(REJECTION_CAUSE_DESC.keys()) + 1),
        "yTicksLabels" : REJECTION_CAUSE_DESC.keys(), 

        "Marker" : ".",
        "LineWidth" : 0,
        "Grid" : 1,

        "ColorBar" : "nipy_spectral",
        "ColorBarLabel" : "GPS PRN",
        "ColorBarMin" : min(gps_prn),
        "ColorBarMax" : max(gps_prn),
        "ColorBarSetTicks": sorted(gps_prn),
        "ColorBarBins": len(gps_prn), 

        "s" : 20, 
        "Label" : 0,
        
        "Annotations": {0: aggregated_dataGPS[PreproIdx["PRN"]]},

        "xData": {0: aggregated_dataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H},
        "yData": {0: aggregated_dataGPS[PreproIdx["REJECT"]]},
        "zData" : {0: [int(convert_satlabel_to_prn(prn)) for prn in aggregated_dataGPS[PreproIdx["PRN"]]]}, 

        "Path": sys.argv[1] + '/OUT/PPVE/SAT/' + 'GPS_REJECTION_FLAGS_s6an_D011Y24.png',
    }

    all_confs = [PlotConfGalileo, PlotConfGPS]

    for conf in all_confs:
        generatePlot(conf)

# Plot Rates
def plotRates(PreproObsFile, PreproObsData, PlotTitle, PlotLabel):
    valid_column = PreproObsData[PreproIdx["VALID"]]
    if PlotLabel == "CODE_RATE":
        
        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["CODE_RATE"]] & valid_column == 1 ]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["CODE_RATE"]] & valid_column == 1 ] 

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Code Rate [m/s]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["CODE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)

        PlotConfGalileoZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Code Rate [m/s]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["CODE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileoZoomed = initPlot(PreproObsFile, PlotConfGalileoZoomed, "GAL " + PlotTitle, "GAL_ZOOMED_" + PlotLabel)

        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate [m/s]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["CODE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        PlotConfGPSZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate [m/s]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["CODE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPSZoomed = initPlot(PreproObsFile, PlotConfGPSZoomed, "GPS " + PlotTitle, "GPS_ZOOMED_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGalileoZoomed, PlotConfGPS, PlotConfGPSZoomed]

        for conf in all_confs:
            generatePlot(conf)

    elif PlotLabel == "PHASE_RATE":

        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["PHASE_RATE"]] & valid_column == 1 ]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["PHASE_RATE"]] & valid_column == 1 ]

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Phase Rate [m/s]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["PHASE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)

        PlotConfGalileoZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Phase Rate [m/s]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["PHASE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileoZoomed = initPlot(PreproObsFile, PlotConfGalileoZoomed, "GAL " + PlotTitle, "GAL_ZOOMED_" + PlotLabel)

        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Phase Rate [m/s]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["PHASE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        PlotConfGPSZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate [m/s]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [-8000, 8000],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["PHASE_RATE"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPSZoomed = initPlot(PreproObsFile, PlotConfGPSZoomed, "GPS " + PlotTitle, "GPS_ZOOMED_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGalileoZoomed, PlotConfGPS, PlotConfGPSZoomed]

        for conf in all_confs:
            generatePlot(conf)
    
    elif PlotLabel == "CODE_RATE_STEP":
        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["CODE_RATE_STEP"]] & valid_column == 1 ]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["CODE_RATE_STEP"]] & valid_column == 1 ]

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Code Rate Step [m/s2]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["CODE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)

        PlotConfGalileoZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Code Rate Step [m/s2]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["CODE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileoZoomed = initPlot(PreproObsFile, PlotConfGalileoZoomed, "GAL " + PlotTitle, "GAL_ZOOMED_" + PlotLabel)

        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate Step [m/s2]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["CODE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        PlotConfGPSZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate Step [m/s2]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["CODE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPSZoomed = initPlot(PreproObsFile, PlotConfGPSZoomed, "GPS " + PlotTitle, "GPS_ZOOMED_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGalileoZoomed, PlotConfGPS, PlotConfGPSZoomed]

        for conf in all_confs:
            generatePlot(conf)

    elif PlotLabel == "PHASE_RATE_STEP":

        PreproObsDataGalileo = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("E") & PreproObsData[PreproIdx["PHASE_RATE_STEP"]] & valid_column == 1 ]
        PreproObsDataGPS = PreproObsData[PreproObsData[PreproIdx["PRN"]].str.startswith("G") & PreproObsData[PreproIdx["PHASE_RATE_STEP"]] & valid_column == 1 ]

        PlotConfGalileo = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Phase Rate Step [m/s2]",

            "xTicks": range(round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGalileo[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGalileo[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["PHASE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileo = initPlot(PreproObsFile, PlotConfGalileo, "GAL " + PlotTitle, "GAL_" + PlotLabel)

        PlotConfGalileoZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GAL Phase Rate Step [m/s2]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : "|",
            "LineWidth" : 1,

            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGalileo[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGalileo[PreproIdx["PHASE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGalileo[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGalileoZoomed = initPlot(PreproObsFile, PlotConfGalileoZoomed, "GAL " + PlotTitle, "GAL_ZOOMED_" + PlotLabel)

        PlotConfGPS = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Phase Rate Step [m/s2]",

            "xTicks": range(round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H) + 1),
            "xLim" : [round(PreproObsDataGPS[PreproIdx["SOD"]].min() / GnssConstants.S_IN_H), round(PreproObsDataGPS[PreproIdx["SOD"]].max() / GnssConstants.S_IN_H)],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["PHASE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPS = initPlot(PreproObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel)

        PlotConfGPSZoomed = {
            "Type": "Lines",
            "FigSize" : (8.4, 6.6),

            "yLabel" : "GPS Code Rate Step [m/s]",

            "xTicks": range(0, 3),
            "xLim" : [0, 2],

            "yLim" : [0, 15],

            "Grid" : 1,
            
            "Marker" : ".",
            "LineWidth" : 1,
            "s": 1.5,

            "ColorBar": "gnuplot",
            "ColorBarLabel": "Elevation [deg]",
            "ColorBarMin" : 0,
            "ColorBarMax" : 90,
            
            "Label" : {0},
            
            "xData": {
                0: PreproObsDataGPS[PreproIdx["SOD"]] / GnssConstants.S_IN_H,
            },

            "yData": {
                0: PreproObsDataGPS[PreproIdx["PHASE_RATE_STEP"]],  
            },

            "zData":{
                0: PreproObsDataGPS[PreproIdx["ELEV"]],                            
            },
        }

        PlotConfGPSZoomed = initPlot(PreproObsFile, PlotConfGPSZoomed, "GPS " + PlotTitle, "GPS_ZOOMED_" + PlotLabel)

        all_confs = [PlotConfGalileo, PlotConfGalileoZoomed, PlotConfGPS, PlotConfGPSZoomed]

        for conf in all_confs:
            generatePlot(conf)


def generatePreproPlots(PreproObsFile):
    
    # Purpose: generate output plots regarding Preprocessing results

    # Parameters
    # ==========
    # PreproObsFile: str
    #         Path to PREPRO OBS output file

    # Returns
    # =======
    # Nothing


    # Satellite Visibility
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["PRN"],PreproIdx["STATUS"],PreproIdx["ELEV"]])
    
    print('INFO: Plot Satellite Visibility Periods ...')

    # Configure plot and call plot generation function
    plotSatVisibility(PreproObsFile, PreproObsData)


    # Number of satellites
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["PRN"],PreproIdx["STATUS"]])
    
    print('INFO: Plot Number of Satellites ...')

    # Configure plot and call plot generation function
    plotNumSats(PreproObsFile, PreproObsData)


    # Code IF - Code IF Smoothed
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["REJECT"],PreproIdx["STATUS"],PreproIdx["ELEV"],\
    PreproIdx["PRN"],PreproIdx["CODE_IF"],PreproIdx["SMOOTH_IF"]])
    
    print('INFO: Plot Code IF - Code IF Smoothed ...')

    # Configure plot and call plot generation function
    plotIFIFSmoothed(PreproObsFile, PreproObsData)


    # C/N0
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["S1"],PreproIdx["S2"],\
        PreproIdx["ELEV"], PreproIdx["PRN"]])
    
    print('INFO: Plot C/N0...')

    # Configure plot and call plot generation function
    plotCN0(PreproObsFile, PreproObsData, 'CN0_F1', 'S1')

    # Configure plot and call plot generation function
    plotCN0(PreproObsFile, PreproObsData, 'CN0_F2', 'S2')


    # Rejection Flags
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["PRN"],PreproIdx["REJECT"]])
    
    print('INFO: Plot Rejection Flags ...')

    # Configure plot and call plot generation function
    plotRejectionFlags(PreproObsFile, PreproObsData)


    # Code Rate
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["VALID"],PreproIdx["CODE_RATE"],\
        PreproIdx["ELEV"], PreproIdx["PRN"]])
    
    print('INFO: Plot Code Rate ...')

    # Configure plot and call plot generation function
    plotRates(PreproObsFile, PreproObsData, 'Code Rate', 'CODE_RATE')


    # Phase Rate
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["VALID"],PreproIdx["PHASE_RATE"],\
        PreproIdx["ELEV"], PreproIdx["PRN"]])

    print('INFO: Plot Phase Rate ...')

    # Configure plot and call plot generation function
    plotRates(PreproObsFile, PreproObsData, 'Phase Rate', 'PHASE_RATE')
    

    # Code Rate Step
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["VALID"],PreproIdx["CODE_RATE_STEP"],\
        PreproIdx["ELEV"], PreproIdx["PRN"]])
    
    print('INFO: Plot Code Rate Step...')

    # Configure plot and call plot generation function
    plotRates(PreproObsFile, PreproObsData, 'Code Rate Step', 'CODE_RATE_STEP')


    # Phase Rate Step
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    PreproObsData = read_csv(PreproObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[PreproIdx["SOD"],PreproIdx["VALID"],PreproIdx["PHASE_RATE_STEP"],\
        PreproIdx["ELEV"], PreproIdx["PRN"]])
    
    print('INFO: Plot Phase Rate Step...')

    # Configure plot and call plot generation function
    plotRates(PreproObsFile, PreproObsData, 'Phase Rate Step', 'PHASE_RATE_STEP')