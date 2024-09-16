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
from collections import OrderedDict
from yaml import dump
from COMMON import GnssConstants as Const
from InputOutput import readConf
from InputOutput import processConf
from InputOutput import createOutputFile
from InputOutput import openInputFile
from InputOutput import readObsEpoch
from InputOutput import generatePreproFile, generateCorrFile
from InputOutput import readLeoPos
from InputOutput import readLeoQuat
from InputOutput import readSatPos
from InputOutput import readSatApo
from InputOutput import readSatClk
from InputOutput import readSatBia
from InputOutput import ObsIdxP
from InputOutput import generateCorrFile
from InputOutput import PreproHdr, CorrHdr
from InputOutput import CSNEPOCHS, CSNPOINTS
from Preprocessing import runPreprocessing
from PreprocessingPlots import generatePreproPlots
from COMMON.Dates import convertJulianDay2YearMonthDay
from COMMON.Dates import convertYearMonthDay2Doy
from Corrections import runCorrectMeas

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
print( '--> RUNNING SENTUS:')
print( '------------------------------------')

# Loop over Julian Days in simulation
#-----------------------------------------------------------------------
for Jd in range(Conf["INI_DATE_JD"], Conf["END_DATE_JD"] + 1):
    # Compute Year, Month and Day in order to build input file name
    Year, Month, Day = convertJulianDay2YearMonthDay(Jd)

    # Compute the Day of Year (DoY)
    Doy = convertYearMonthDay2Doy(Year, Month, Day)

    # Display Message
    print( '\n*** Processing Day of Year: ' + str(Doy) + ' ... ***')

    # Define the full path and name to the OBS INFO file to read
    ObsFile = Scen + \
        '/INP/OBS/' + "OBS_%s_Y%02dD%03d.dat" % \
            (Conf['SAT_ACRONYM'], Year % 100, Doy)

    # Define the full path and name to the Sentinel POS file to read and open the file
    SatPosFile = Scen + \
        '/INP/SP3/' + "LEO_POS_%s_Y%02dD%03d.dat" % \
            (Conf['SAT_ACRONYM'], Year % 100, Doy)
    # Display Message
    print("INFO: Reading file: %s..." %
    SatPosFile)
    # Read the file
    LeoPosInfo = readLeoPos(SatPosFile)
    
    # Define the full path and name to the Sentinel Quaternions file to read and open the file
    SatQuatFile = Scen + \
        '/INP/ATT/' + "LEO_QUATERNIONS_%s_Y%02dD%03d.dat" % \
            (Conf['SAT_ACRONYM'], Year % 100, Doy)
    # Display Message
    print("INFO: Reading file: %s..." %
    SatQuatFile)
    # Read the file
    LeoQuatInfo = readLeoQuat(SatQuatFile)

    # Define the full path and name to the SAT_POS file to read and open the file
    SatPosFile = Scen + \
        '/INP/SP3/' + "SAT_POS_CODE_Y%02dD%03d.dat" % \
            (Year % 100, Doy)
    # Display Message
    print("INFO: Reading file: %s..." %
    SatPosFile)
    # Read the file
    SatPosInfo = readSatPos(SatPosFile)

    # Define the full path and name to the SAT_APO file to read and open the file
    SatApoFile = Scen + \
        '/INP/ATX/' + Conf["SAT_APO_FILE"]
    # Display Message
    print("INFO: Reading file: %s..." %
    SatApoFile)
    # Read the file
    SatApoInfo = readSatApo(SatApoFile)

    # Define the full path and name to the SAT_CLK file to read and open the file
    SatClkFile = Scen + \
        '/INP/CLK/' + "SAT_CLK_CODE_Y%02dD%03d_300S.dat" % \
            (Year % 100, Doy)
    # Display Message
    print("INFO: Reading file: %s..." %
    SatClkFile)
    # Read the file
    SatClkInfo = readSatClk(SatClkFile)

    # Define the full path and name to the SAT_BIA file to read and open the file
    SatBiaFile = Scen + \
        '/INP/BIA/' + Conf["SAT_BIA_FILE"]
    # Display Message
    print("INFO: Reading file: %s..." %
    SatBiaFile)
    # Read the file
    SatBiaInfo = readSatBia(SatBiaFile)



    # # If PREPRO outputs are requested
    # if Conf["PREPRO_OUT"] == 1:
    #     # Define the full path and name to the output PREPRO OBS file
    #     PreproObsFile = Scen + \
    #         '/OUT/PPVE/' + "PREPRO_OBS_%s_Y%02dD%03d.dat" % \
    #             (Conf['SAT_ACRONYM'], Year % 100, Doy)

    #     # Display Message
    #     print("INFO: Reading file: %s and generating PREPRO figures..." %
    #     PreproObsFile)

    #     # Generate Preprocessing plots
    #     generatePreproPlots(PreproObsFile)

    # sys.exit()

    # If Preprocessing outputs are activated
    if Conf["PREPRO_OUT"] == 1:
        # Define the full path and name to the output PREPRO OBS file
        PreproObsFile = Scen + \
            '/OUT/PPVE/' + "PREPRO_OBS_%s_Y%02dD%03d.dat" % \
                (Conf['SAT_ACRONYM'], Year % 100, Doy)

        # Create output file
        fpreprobs = createOutputFile(PreproObsFile, PreproHdr)

    # If Corrected outputs are activated
    if Conf["CORR_OUT"] == 1:
        # Define the full path and name to the output CORR file
        CorrObsFile = Scen + \
            '/OUT/CORR/' + "CORR_OBS_%s_Y%02dD%03d.dat" % \
                (Conf['SAT_ACRONYM'], Year % 100, Doy)

        # Create output file
        fcorr = createOutputFile(CorrObsFile, CorrHdr)
        
    # Initialize Variables
    EndOfFile = False
    ObsInfo = [None]
    PrevPreproObsInfo = {}
    # Declare Sod_1 and SatComPos_1 
    Sod_1 = {}
    SatComPos_1 = {}
    for const in ['G', 'E']:
        for prn in range(1, Const.MAX_NUM_SATS_CONSTEL + 1):
            PrevPreproObsInfo["%s%02d" % (const,prn)] = {
                "PrevEpoch": 86400,                                          # Previous SoD

                "ResetHatchFilter": 1,                                       # Flag to reset Hatch filter
                "Ksmooth": 0,                                                # Hatch filter K
                "PrevSmooth": 0,                                             # Previous Smooth Observable
                "IF_P_Prev": 0,                                              # Previous IF of the phases
                
                "PrevL1": Const.NAN,                                         # Previous L1
                "PrevPhaseRateL1": Const.NAN,                                # Previous Phase Rate
                "PrevC1": Const.NAN,                                         # Previous C1
                "PrevRangeRateL1": Const.NAN,                                # Previous Code Rate
                
                "PrevL2": Const.NAN,                                         # Previous L1
                "PrevPhaseRateL2": Const.NAN,                                # Previous Phase Rate
                "PrevC2": Const.NAN,                                         # Previous C2
                "PrevRangeRateL2": Const.NAN,                                # Previous Code Rate

                "CycleSlipBuffIdx": 0,                                         # Index of CS buffer
                "CycleSlipFlagIdx": 0,                                         # Index of CS flag array
                "GF_L_Prev": [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS]),      # Array with previous GF carrier phase observables
                "GF_Epoch_Prev": [0.0] * int(Conf["CYCLE_SLIPS"][CSNPOINTS]),  # Array with previous epochs
                "CycleSlipFlags": [0.0] * int(Conf["CYCLE_SLIPS"][CSNEPOCHS]), # Array with last cycle slips flags
                "CycleSlipDetectFlag": 0,                                      # Flag indicating if a cycle slip has been detected

                "PrealignOffset": 0,                                           # Phase prealignment offset

            } # End of SatPreproObsInfo
            # Populate Sod_1 and SatComPos_1 with PRN 
            Sod_1["%s%02d" % (const,prn)] = []
            SatComPos_1["%s%02d" % (const,prn)] = []


    # Display Message
    print("INFO: Reading file: %s..." %
    ObsFile)

    # Open OBS file
    with open(ObsFile, 'r') as fobs:

        # LOOP over all Epochs of OBS file
        # ----------------------------------------------------------
        while not EndOfFile:

            # If ObsInfo is not empty
            if ObsInfo != []:

                # Read Only One Epoch
                ObsInfo = readObsEpoch(fobs)

                # If ObsInfo is empty, exit loop
                if ObsInfo == []:
                    break

                # Preprocess OBS measurements
                # ----------------------------------------------------------
                PreproObsInfo = runPreprocessing(Conf, ObsInfo, PrevPreproObsInfo)

                # If PREPRO outputs are requested
                if Conf["PREPRO_OUT"] == 1:
                    # Generate output file
                    generatePreproFile(fpreprobs, PreproObsInfo)

                # Get SoD
                Sod = int(ObsInfo[1][0][ObsIdxP["SOD"]])

                # The rest of the analyses are executed every configured sampling rate
                if(Sod % Conf["SAMPLING_RATE"] == 0):
                    # Correct measurements and estimate the variances
                    # ----------------------------------------------------------
                    CorrInfo, RcvrRefPosXyz, RcvrRefPosLlh = runCorrectMeas(Year,
                                                                            Doy,
                                                                            Conf, 
                                                                            PreproObsInfo, 
                                                                            LeoPosInfo,
                                                                            LeoQuatInfo,
                                                                            SatPosInfo,
                                                                            SatApoInfo,
                                                                            SatClkInfo,
                                                                            SatBiaInfo,
                                                                            SatComPos_1,
                                                                            Sod_1
                                                                            )

                    # If CORR outputs are requested
                    if Conf["CORR_OUT"] == 1:
                        # Generate output file
                        generateCorrFile(fcorr, CorrInfo)

    # If PREPRO outputs are requested
    if Conf["PREPRO_OUT"] == 1:
        # Close PREPRO output file
        fpreprobs.close()

        # Display Message
        print("INFO: Reading file: %s and generating PREPRO figures..." %
        PreproObsFile)

        # Generate Preprocessing plots
        generatePreproPlots(PreproObsFile)

    # If CORR outputs are requested
    # if Conf["CORR_OUT"] == 1:
        # Close CORR output file

# End of JD loop

print( '\n------------------------------------')
print( '--> END OF SENTUS ANALYSIS')
print( '------------------------------------')

#######################################################
# End of Sentus.py
#######################################################
