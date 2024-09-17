#!/usr/bin/env python

########################################################################
# Sentus.py:
# This is the Main Module of SENTUS tool
#
#  Project:        SENTUS
#  File:           Sentus.py
#
#   Author: Agustin Pernigotti
#   Copyright 2024 Agustin Pernigotti
#
# Usage:
#   Sentus.py $SCEN_PATH
########################################################################

import sys, os

# Update Path to reach COMMON
Common = os.path.dirname(
    os.path.abspath(sys.argv[0])) + '/COMMON'
sys.path.insert(0, Common)

# Import External and Internal functions and Libraries
#----------------------------------------------------------------------
from InputOutput import readConf
from InputOutput import processConf
from COMMON.Dates import convertJulianDay2YearMonthDay
from COMMON.Dates import convertYearMonthDay2Doy
from CorrectionsPlots import generateCorrPlots

#----------------------------------------------------------------------
# INTERNAL FUNCTIONS
#----------------------------------------------------------------------

def displayUsage():
    sys.stderr.write("ERROR: Please provide path to SCENARIO as a unique argument\n")

#######################################################
# MAIN BODY
#######################################################

# Check InputOutput Arguments
if len(sys.argv) != 2:
    displayUsage()
    sys.exit()

# Extract the arguments
Scen = sys.argv[1]

# Select the Configuratiun file name
CfgFile = Scen + '/CFG/sentus.cfg'

# Read conf file
Conf = readConf(CfgFile)

# Process Configuration Parameters
Conf = processConf(Conf)

# Print header
print( '------------------------------------')
print( '--> RUNNING SENTUS Plots:')
print( '------------------------------------')

# Loop over Julian Days in simulation
#-----------------------------------------------------------------------
for Jd in range(Conf["INI_DATE_JD"], Conf["END_DATE_JD"] + 1):
    # Compute Year, Month and Day in order to build input file name
    Year, Month, Day = convertJulianDay2YearMonthDay(Jd)

    # Compute the Day of Year (DoY)
    Doy = convertYearMonthDay2Doy(Year, Month, Day)

    # If Corrected outputs are activated
    if Conf["CORR_OUT"] == 1:
        # Define the full path and name to the output CORR file
        CorrObsFile = Scen + \
            '/OUT/CORR/' + "CORR_OBS_%s_Y%02dD%03d.dat" % \
                (Conf['SAT_ACRONYM'], Year % 100, Doy)


    # If CORR outputs are requested
    if Conf["CORR_OUT"] == 1:

        # Display Message
        print("INFO: Reading file: %s and generating CORR figures..." %
        CorrObsFile)

        # Generate Preprocessing plots
        generateCorrPlots(CorrObsFile)

# End of JD loop

print( '\n------------------------------------')
print( '--> END OF SENTUS Plots')
print( '------------------------------------')

#######################################################
# End of Sentus.py
#######################################################
