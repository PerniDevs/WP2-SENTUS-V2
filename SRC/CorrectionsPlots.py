#!/usr/bin/env python

########################################################################
# PreprocessingPlots.py:
# This is the CorrectionsPlots Module of SENTUS tool
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
import numpy as np
from pandas import unique
from pandas import read_csv
from InputOutput import CorrIdx
from InputOutput import REJECTION_CAUSE_DESC
sys.path.append(os.getcwd() + '/' + \
    os.path.dirname(sys.argv[0]) + '/' + 'COMMON')
from COMMON import GnssConstants
from COMMON.Plots import generatePlot
from COMMON.allPRNs import allprns
from COMMON.Coordinates import xyz2llh


def initPlot(PreproObsFile, PlotConf, Title, Label, xLabelRequired:bool):
    PreproObsFileName = os.path.basename(PreproObsFile)
    PreproObsFileNameSplit = PreproObsFileName.split('_')
    Rcvr = PreproObsFileNameSplit[2]
    DatepDat = PreproObsFileNameSplit[3]
    Date = DatepDat.split('.')[0]
    Year = Date[1:3]
    Doy = Date[4:]

    if xLabelRequired:
        PlotConf["xLabel"] = "Hour of Day %s" % Doy 

    PlotConf["Title"] = "%s from %s on Year %s"\
        " DoY %s" % (Title, Rcvr, Year, Doy)

    PlotConf["Path"] = sys.argv[1] + '/OUT/CORR/figures/' + \
        '%s_%s_Y%sD%s.png' % (Label, Rcvr, Year, Doy)
    
    return PlotConf


# Function to convert 'G01', 'G02', etc. to 1, 2, etc.
def convert_satlabel_to_prn(value):
    return int(value[1:])


# Function to convert 'G01', 'G02', etc. to 'G'
def convert_satlabel_to_const(value):
    return value[0]


def plotSatTrack(CorrObsFile, CorrObsData):
    
    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    maxLat = 60
    maxLon = 180

    # Transform ECEF to Geodetic GALILEO
    xGalData = GalData[CorrIdx["SAT-X"]].to_numpy()
    yGalData = GalData[CorrIdx["SAT-Y"]].to_numpy()
    zGalData = GalData[CorrIdx["SAT-Z"]].to_numpy()
    DataLen = len(GalData[CorrIdx["SAT-X"]])
    LongitudeGal = np.zeros(DataLen)
    LatitudeGal = np.zeros(DataLen)
    # transformer = Transformer.from_crs('epsg:4978', 'epsg:4326')
    for index in range(DataLen):
        x = xGalData[index]
        y = yGalData[index]
        z = zGalData[index]
        LongitudeGal[index], LatitudeGal[index], h = xyz2llh(x, y, z)

    # Transform ECEF to Geodetic GPS
    xGPSData = GpsData[CorrIdx["SAT-X"]].to_numpy()
    yGPSData = GpsData[CorrIdx["SAT-Y"]].to_numpy()
    zGPSData = GpsData[CorrIdx["SAT-Z"]].to_numpy()
    DataLen = len(GpsData[CorrIdx["SAT-X"]])
    LongitudeGPS = np.zeros(DataLen)
    LatitudeGPS = np.zeros(DataLen)
    # transformer = Transformer.from_crs('epsg:4978', 'epsg:4326')
    for index in range(DataLen):
        x = xGPSData[index]
        y = yGPSData[index]
        z = zGPSData[index]
        LongitudeGPS[index], LatitudeGPS[index], h = xyz2llh(x, y, z)

    PlotTitle = "Satellite Tracks"
    PlotLabel = "SATELLITE_TRACKS"

    PlotConfGal = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "LonMin" : -maxLon,
        "LonMax" : maxLon,
        "LatMin" : -maxLat,
        "LatMax" : maxLat,
        "LonStep" : 15,
        "LatStep" : 10,

        # PlotConf["yLabel"] = "Latitude [deg]"
        "yTicks" : range(-maxLat,maxLat+1,10),
        "yLim" : [-maxLat, maxLat],

        # PlotConf["xLabel"] = "Longitude [deg]"
        "xTicks" : range(-maxLon,maxLon+1,15),
        "xLim" : [-maxLon, maxLon],

        "Grid": True,
        "s" : 1,

        "Map": True,
        
        "Marker" : '.',
        "LineWidth" : 0.1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : LongitudeGal
            },
        "yData":{
            0 : LatitudeGal
            },
        "zData":{
            0 : GalData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGal = initPlot(CorrObsFile, PlotConfGal, "Galileo " + PlotTitle, "GAL_" + PlotLabel, xLabelRequired=False)

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "LonMin" : -maxLon,
        "LonMax" : maxLon,
        "LatMin" : -maxLat,
        "LatMax" : maxLat,
        "LonStep" : 15,
        "LatStep" : 10,

        # PlotConf["yLabel"] = "Latitude [deg]"
        "yTicks" : range(-maxLat,maxLat+1,10),
        "yLim" : [-maxLat, maxLat],

        # PlotConf["xLabel"] = "Longitude [deg]"
        "xTicks" : range(-maxLon,maxLon+1,15),
        "xLim" : [-maxLon, maxLon],

        "Grid": True,
        "s" : 1,

        "Map": True,

        "Marker" : '.',
        "LineWidth" : 0.1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : LongitudeGPS
            },
        "yData":{
            0 : LatitudeGPS
            },
        "zData":{
            0 : GpsData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGPS = initPlot(CorrObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel, xLabelRequired=False)

    allConfs = [PlotConfGal,PlotConfGPS]
    for conf in allConfs:
        generatePlot(conf)
    

def plotFlightTime(CorrObsFile, CorrObsData):

    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    PlotTitle = "Flight Time"
    PlotLabel = "FLIGHT_TIME"

    PlotConfGal = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Fligh Time [miliseconds]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : GalData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GalData[CorrIdx["FLIGHT-TIME"]]
            },
        "zData":{
            0 : GalData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGal = initPlot(CorrObsFile, PlotConfGal, "Galileo " + PlotTitle, "GAL_" + PlotLabel, xLabelRequired=True)

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Fligh Time [miliseconds]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : GpsData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GpsData[CorrIdx["FLIGHT-TIME"]]
            },
        "zData":{
            0 : GpsData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGPS = initPlot(CorrObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel, xLabelRequired=True)
    
    allConfs = [PlotConfGal,PlotConfGPS]
    for conf in allConfs:
        generatePlot(conf)


def plotDtr(CorrObsFile, CorrObsData):

    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    PlotTitle = "Relativistic Correction (DTR)"
    PlotLabel = "DTR"

    PlotConfGal = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Relativistic correction (DTR) [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : GalData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GalData[CorrIdx["DTR"]]
            },
        "zData":{
            0 : GalData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGal = initPlot(CorrObsFile, PlotConfGal, "Galileo " + PlotTitle, "GAL_" + PlotLabel, xLabelRequired=True)

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Relativistic correction (DTR) [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "Elevation [deg]",
        "ColorBarMin" : 0.,
        "ColorBarMax" : 90.,

        "Label": {0},
        "xData":{
            0 : GpsData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GpsData[CorrIdx["DTR"]]
            },
        "zData":{
            0 : GpsData[CorrIdx["ELEV"]]
            },
    }

    PlotConfGPS = initPlot(CorrObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel, xLabelRequired=True)

    allConfs = [PlotConfGal,PlotConfGPS]
    for conf in allConfs:
        generatePlot(conf)


def plotCodeResiduals(CorrObsFile, CorrObsData):
    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    PlotTitle = "Code Residuals"
    PlotLabel = "CODE_RESIDUALS"

    PlotConfGal = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Code Residuals [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "PRN",
        "ColorBarMin" : GalData[CorrIdx["PRN"]].min(),
        "ColorBarMax" : GalData[CorrIdx["PRN"]].max(),
        "ColorBarSetTicks" : sorted(unique(GalData[CorrIdx["PRN"]])),

        "Label": {0},
        "xData":{
            0 : GalData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GalData[CorrIdx["CODE-RES"]]
            },
        "zData":{
            0 : GalData[CorrIdx["PRN"]]
            },
    }

    PlotConfGal = initPlot(CorrObsFile, PlotConfGal, "Galileo " + PlotTitle, "GAL_" + PlotLabel, xLabelRequired=True)

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Code Residuals [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "PRN",
        "ColorBarMin" : GpsData[CorrIdx["PRN"]].min(),
        "ColorBarMax" : GpsData[CorrIdx["PRN"]].max(),
        "ColorBarSetTicks" : sorted(unique(GpsData[CorrIdx["PRN"]])),

        "Label": {0},
        "xData":{
            0 : GpsData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GpsData[CorrIdx["CODE-RES"]]
            },
        "zData":{
            0 : GpsData[CorrIdx["PRN"]]
            },
    }

    PlotConfGPS = initPlot(CorrObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel, xLabelRequired=True)

    allConfs = [PlotConfGal,PlotConfGPS]
    for conf in allConfs:
        generatePlot(conf)


def plotPhaseResiduals(CorrObsFile, CorrObsData):
    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    PlotTitle = "Phase Residuals"
    PlotLabel = "PHASE_RESIDUALS"

    PlotConfGal = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Phase Residuals [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "PRN",
        "ColorBarMin" : GalData[CorrIdx["PRN"]].min(),
        "ColorBarMax" : GalData[CorrIdx["PRN"]].max(),
        "ColorBarSetTicks" : sorted(unique(GalData[CorrIdx["PRN"]])),

        "Label": {0},
        "xData":{
            0 : GalData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GalData[CorrIdx["PHASE-RES"]]
            },
        "zData":{
            0 : GalData[CorrIdx["PRN"]]
            },
    }

    PlotConfGal = initPlot(CorrObsFile, PlotConfGal, "Galileo " + PlotTitle, "GAL_" + PlotLabel, xLabelRequired=True)

    PlotConfGPS = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Phase Residuals [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        "s" : 1,
        
        "Marker" : '.',
        "LineWidth" : 1,

        "ColorBar" : "gnuplot",
        "ColorBarLabel" : "PRN",
        "ColorBarMin" : GpsData[CorrIdx["PRN"]].min(),
        "ColorBarMax" : GpsData[CorrIdx["PRN"]].max(),
        "ColorBarSetTicks" : sorted(unique(GpsData[CorrIdx["PRN"]])),

        "Label": {0},
        "xData":{
            0 : GpsData[CorrIdx["SOD"]] / GnssConstants.S_IN_H
            },
        "yData":{
            0 : GpsData[CorrIdx["PHASE-RES"]]
            },
        "zData":{
            0 : GpsData[CorrIdx["PRN"]]
            },
    }

    PlotConfGPS = initPlot(CorrObsFile, PlotConfGPS, "GPS " + PlotTitle, "GPS_" + PlotLabel, xLabelRequired=True)

    allConfs = [PlotConfGal,PlotConfGPS]
    for conf in allConfs:
        generatePlot(conf)


def plotRcvrClk(CorrObsFile, CorrObsData):

    GalData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("E")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]
    GpsData = CorrObsData[(CorrObsData[CorrIdx["CONST"]].str.startswith("G")) & (CorrObsData[CorrIdx["FLAG"]] == 1)]

    PlotTitle = "Receiver Clock Estimation"
    PlotLabel = "RCVR_CLK"
    
    PlotConf = {
        "Type": "Lines",
        "FigSize" : (14.4, 10.6),

        "yLabel": "Receiver Clock [m]",

        "xTicks" : range(0, 25),
        "xLim" : [0, 24],

        "Grid": True,
        
        "c" : {
            0: "red",
            1: "blue",
        },
        
        "Marker" : '',
        "LineWidth" : 1,
        "LineStyle" : '-',


        "Label": {
            0:"GAL",
            1:"GPS",
            },

        "LabelLoc": "best",

        "xData":{
            0 : GalData[CorrIdx["SOD"]] / GnssConstants.S_IN_H,
            1 : GpsData[CorrIdx["SOD"]] / GnssConstants.S_IN_H,
            },

        "yData":{
            0 : GalData[CorrIdx["RCVR-CLK"]],
            1 : GpsData[CorrIdx["RCVR-CLK"]]
            },

    }

    PlotConf = initPlot(CorrObsFile, PlotConf, PlotTitle, PlotLabel, xLabelRequired=True)

    generatePlot(PlotConf)



def generateCorrPlots(CorrObsFile):
    
    # Purpose: generate output plots regarding Corrections results

    # Parameters
    # ==========
    # CorrObsFile: str
    #         Path to CORR OBS output file

    # Returns
    # =======
    # Nothing


    # Satellite Tracks
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"],
             CorrIdx["ELEV"], CorrIdx["SAT-X"], CorrIdx["SAT-Y"], CorrIdx["SAT-Z"]])
    
    print('INFO: Plot Satellite Tracks...')

    # Configure plot and call plot generation function
    plotSatTrack(CorrObsFile, CorrObsData)

    # Flight time
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"],
             CorrIdx["ELEV"], CorrIdx["FLIGHT-TIME"]])
    
    print('INFO: Plot Satellite Time of Flight...')

    # Configure plot and call plot generation function
    plotFlightTime(CorrObsFile, CorrObsData)


    # DTR
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"],
             CorrIdx["ELEV"], CorrIdx["DTR"]])
    
    print('INFO: Plot DTR...')

    # Configure plot and call plot generation function
    plotDtr(CorrObsFile, CorrObsData)


    # Code Residuals
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"],
             CorrIdx["ELEV"], CorrIdx["CODE-RES"]])
    
    print('INFO: Plot Code Residuals...')

    # Configure plot and call plot generation function
    plotCodeResiduals(CorrObsFile, CorrObsData)


    # Phase Residuals
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"],
             CorrIdx["ELEV"], CorrIdx["PHASE-RES"]])
    
    print('INFO: Plot Phase Residuals...')

    # Configure plot and call plot generation function
    plotPhaseResiduals(CorrObsFile, CorrObsData)


    # Receiver Clock
    # ----------------------------------------------------------
    # Read the cols we need from PREPRO OBS file
    CorrObsData = read_csv(CorrObsFile, delim_whitespace=True, skiprows=1, header=None,\
    usecols=[CorrIdx["SOD"],CorrIdx["CONST"],CorrIdx["PRN"],CorrIdx["FLAG"], CorrIdx["RCVR-CLK"]])
    
    print('INFO: Plot Receiver Clock...')

    # Configure plot and call plot generation function
    plotRcvrClk(CorrObsFile, CorrObsData)