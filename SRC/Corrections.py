#!/usr/bin/env python

########################################################################
# PETRUS/SRC/Corrections.py:
# This is the Corrections Module of SENTUS tool
#
#  Project:        SENTUS
#  File:           Corrections.py
#  Date(YY/MM/DD): 18/09/2024
#
#   Author: Agustin Pernigotti
#   Copyright 2024 Agustin Pernigotti
#
# -----------------------------------------------------------------
# Date       | Author             | Action
# -----------------------------------------------------------------
#
########################################################################

# Import External and Internal functions and Libraries
#----------------------------------------------------------------------
import sys, os
# Add path to find all modules
Common = os.path.dirname(os.path.dirname(
    os.path.abspath(sys.argv[0]))) + '/COMMON'
sys.path.insert(0, Common)
from collections import OrderedDict
from COMMON import GnssConstants as Const
from COMMON.Misc import findSun, crossProd
from COMMON.LeoComputation import computeLeoComPos, computeSatClkBias, computeRcvrApo, computeSatComPos, computeFlightTime, computeSagnacCorr, computeSatApo, getBiases, computeDtr, getUere, computeGeomRange, estimateRcvrClk
import numpy as np

# Import Dictionaries indices
from InputOutput import ObsIdxC, ObsIdxP, LeoPosIdx, LeoQuatIdx, SatPosIdx, SatApoIdx, SatClkIdx, SatBiaIdx 


def runCorrectMeas(Year,
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
                   ):

    # Purpose: correct GNSS preprocessed measurements and compute the first
    #          pseudo range residuals

    #          More in detail, this function handles the following:
    #          tasks:

    #             *  Compute the Satellite Antenna Phase Center position at the transmission time and corrected from the Sagnac
    #                effect interpolating the SP3 file positions
    #             *  Compute the Satellite Clock Bias interpolating the biases coming from the RINEX CLK file and
    #                applying the Relativistic Correction (DTR)
    #             *  Correct the Pre-processed measurements from Geometrical Range, Satellite clock and Troposphere. 
    #             *  Build the Corrected Measurements and Measurement Residuals
    #             *  Build the Sigma UERE


    # Parameters
    # ==========
    # Conf: dict
    #         Configuration dictionary
    # Rcvr: list
    #         Receiver information: position, masking angle...
    # ObsInfo: list
    #         OBS info for current epoch
    #         ObsInfo[1][1] is the second field of the 
    #         second satellite
    # PreproObsInfo: dict
    #         Preprocessed observations for current epoch per sat
    #         PreproObsInfo["G01"]["C1"]
    # LeoPosInfo: dict
    #         containing the LEO reference positions
    # LeoQuatInfo: dict
    #         containing the LEO quaternions
    # SatPosInfo: dict
    #         containing the SP3 file info
    # SatApoInfo: dict
    #         containing the ANTEX file info
    # SatClkInfo: dict
    #         containing the RINEX CLK file info
    # SatBiaInfo: dict
    #         containing the BIA file info
    # SatComPos_1: dict
    #         containing the previous satellite positions
    # Sod_1: dict
    #         containing the time stamp of previous satellite positions

    # Returns
    # =======
    # CorrInfo: dict
    #         Corrected measurements for current epoch per sat
    #         CorrInfo["G01"]["CorrectedPsr"]

    # Initialize output
    CorrInfo = OrderedDict({})
    RcvrRefPosXyz = np.zeros(3)
    RcvrRefPosLlh = np.zeros(3)

    RcvrRefPosXyzCom = computeLeoComPos(LeoPosInfo, Const.KM_TO_M)

    ResidualsAccumGPS = []
    ResidualsAccumGAL = []
    UeresAccumGPS = []
    UeresAccumGAL = []

    # Loop over satellites
    for SatLabel, SatPrepro in PreproObsInfo.items():
        # Get constellation
        Constel = SatLabel[0]
        
        Wave = {}
        # Get wavelengths
        if Constel == 'G':

            # L1 wavelength
            Wave["F1"] = Const.GPS_L1_WAVE

            # L2 wavelength
            Wave["F2"] = Const.GPS_L2_WAVE

            # Gamma GPS
            GammaF1F2 = Const.GPS_GAMMA_L1L2

        elif Constel == 'E':

            # E1 wavelength
            Wave["F1"] = Const.GAL_E1_WAVE

            # E5a wavelength
            Wave["F2"] = Const.GAL_E5A_WAVE

            # Gamma Galileo
            GammaF1F2 = Const.GAL_GAMMA_E1E5A

        # Initialize output info
        SatCorrInfo = {
            "Sod": 0.0,             # Second of day
            "Doy": 0,               # Day of year
            "Elevation": 0.0,       # Elevation
            "Azimuth": 0.0,         # Azimuth
            "Flag": 1,              # 0: Not Used 1: Used
            "LeoX": 0.0,            # X-Component of the Receiver CoP Position 
            "LeoY": 0.0,            # Y-Component of the Receiver CoP Position  
            "LeoZ": 0.0,            # Z-Component of the Receiver CoP Position  
            "LeoApoX": 0.0,         # X-Component of the Receiver APO in ECEF
            "LeoApoY": 0.0,         # Y-Component of the Receiver APO in ECEF
            "LeoApoZ": 0.0,         # Z-Component of the Receiver APO in ECEF
            "SatX": 0.0,            # X-Component of the Satellite CoP Position 
                                    # at transmission time and corrected from Sagnac
            "SatY": 0.0,            # Y-Component of the Satellite CoP Position  
                                    # at transmission time and corrected from Sagnac
            "SatZ": 0.0,            # Z-Component of the Satellite CoP Position  
                                    # at transmission time and corrected from Sagnac
            "SatApoX": 0.0,         # X-Component of the Satellite APO in ECEF
            "SatApoY": 0.0,         # Y-Component of the Satellite APO in ECEF
            "SatApoZ": 0.0,         # Z-Component of the Satellite APO in ECEF
            "ApoProj": 0.0,         # Projection of the Satellite APO
            "SatClk": 0.0,          # Satellite Clock Bias
            "SatCodeBia": 0.0,      # Satellite Code Bias
            "SatPhaseBia": 0.0,     # Satellite Phase Bias
            "FlightTime": 0.0,      # Signal Flight Time
            "Dtr": 0.0,             # Relativistic correction
            "CorrCode": 0.0,        # Code corrected from delays
            "CorrPhase": 0.0,       # Phase corrected from delays
            "GeomRange": 0.0,       # Geometrical Range (distance between Satellite 
                                    # Position and Receiver Reference Position)
            "CodeResidual": 0.0,    # Code Residual
            "PhaseResidual": 0.0,   # Phase Residual
            "RcvrClk": 0.0,         # Receiver Clock estimation
            "SigmaUere": 0.0,       # Sigma User Equivalent Range Error (Sigma of 
                                    # the total residual error associated to the 
                                    # satellite)

        } # End of SatCorrInfo
    
        # Prepare outputs
        # Get Sod
        SatCorrInfo["Sod"] = float(SatPrepro["Sod"])
        # Get Doy
        SatCorrInfo["Doy"] = Doy
        # Get Elevation
        SatCorrInfo["Elevation"] = SatPrepro["Elevation"]
        # Get Azimuth
        SatCorrInfo["Azimuth"] = SatPrepro["Azimuth"]

        # Check if there is lack of previous information
        if Sod_1[SatLabel] != []:
            # Apply corrections only for those satellites with Valid measurements
            if  SatPrepro["Status"] == 1:
                
                # Compute Satellite Clock Bias (linear interpolation between closer inputs)
                # --------------------------------------------------------------------------
                SatClkBias, maxSodSatCLKInfo = computeSatClkBias(SatCorrInfo["Sod"], SatLabel, SatClkInfo)
                
                if (SatCorrInfo["Sod"] <= maxSodSatCLKInfo).any():
                    # Compute DeltaT
                    # --------------------------------------------------------------------------
                    DeltaT = SatPrepro["C1"]/Const.SPEED_OF_LIGHT

                    # Compute Transmission Time
                    # --------------------------------------------------------------------------
                    TransmissionTime = SatCorrInfo["Sod"] - DeltaT - SatClkBias

                    # Compute receiver Position at reception Time
                    # --------------------------------------------------------------------------
                    # Get Receiver APO in ECEF coordinates
                    RcvrApoXyz = computeRcvrApo(Conf, Year, Doy, SatCorrInfo["Sod"], SatLabel, LeoQuatInfo)

                    # Apply the APO
                    # --------------------------------------------------------------------------
                    RcvrRefPosXyz = np.array([(RcvrRefPosXyzCom[RcvrRefPosXyzCom["SOD"] == SatCorrInfo["Sod"]]["xCM"] + RcvrApoXyz[0]).iloc[0],
                                            (RcvrRefPosXyzCom[RcvrRefPosXyzCom["SOD"] == SatCorrInfo["Sod"]]["yCM"] + RcvrApoXyz[1]).iloc[0],
                                            (RcvrRefPosXyzCom[RcvrRefPosXyzCom["SOD"] == SatCorrInfo["Sod"]]["zCM"] + RcvrApoXyz[2]).iloc[0]])                              

                    # Compute Satellite CoM Position at Transmission Time
                    # 10-point Lagrange interpolation between closer inputs (SP3 positions)
                    # -------------------------------------------------------------------------- 
                    SatComPos = computeSatComPos(TransmissionTime, SatLabel, SatPosInfo)
                    
                    # Compute Flight Time
                    # -------------------------------------------------------------------------- 
                    FlightTime = computeFlightTime(SatComPos, RcvrRefPosXyz) * Const.S_TO_MS

                    # Apply Sagnac correction
                    # -------------------------------------------------------------------------- 
                    SatComPosSagnac = computeSagnacCorr(SatComPos, FlightTime)

                    # Compute APO in ECEF from ANTEX APOs in
                    # satellite-body reference frame
                    # -------------------------------------------------------------------------- 
                    SunPos = findSun(Year, Doy, SatCorrInfo["Sod"])
                    Apo = computeSatApo(SatComPosSagnac, SunPos, GammaF1F2, SatApoInfo[(SatApoInfo[SatApoIdx["CONST"]] == SatLabel[0]) & 
                                                                            (SatApoInfo[SatApoIdx["PRN"]] == int(SatLabel[1:]))])

                    # Apply APOs to the satellite Position
                    SatCopPos = SatComPosSagnac + Apo

                    # Get Satellite Biases in meters
                    satCodeBias, satPhaseBias = getBiases(GammaF1F2, SatBiaInfo[(SatBiaInfo[SatBiaIdx["CONST"]] == SatLabel[0]) & 
                                                                    (SatBiaInfo[SatBiaIdx["PRN"]] == int(SatLabel[1:]))])
                    
                    # Compute DTR (Relativistic correction)
                    # -------------------------------------------------------------------------- 
                    Dtr = computeDtr(SatComPos_1[SatLabel][-1], SatComPosSagnac, SatCorrInfo["Sod"], Sod_1[SatLabel][-1])

                    # Apply Dtr to Clock Bias
                    if Dtr != Const.NAN:
                        SatClkBias = SatClkBias * Const.SPEED_OF_LIGHT + Dtr
                    else:
                        SatClkBias = SatClkBias * Const.SPEED_OF_LIGHT

                    # Get Sigma UERE from Conf
                    # -------------------------------------------------------------------------- 
                    UERE = getUere(Conf, SatLabel) if Dtr != Const.NAN else 0

                    # Corrected Measurements from previous information
                    # --------------------------------------------------------------------------
                    CorrCode = (PreproObsInfo[SatLabel]["SmoothIF"] + satCodeBias + SatClkBias ) if Dtr != Const.NAN else 0
                    CorrPhase = (PreproObsInfo[SatLabel]["IF_P"] + satPhaseBias + SatClkBias) if Dtr != Const.NAN else 0

                    # Compute the Geometrical Range
                    # --------------------------------------------------------------------------
                    GeomRange = computeGeomRange(SatCopPos, RcvrRefPosXyz) if Dtr != Const.NAN else 0

                    # Compute the First Residual removing the Geometrical Range
                    # They Include Receiver Clock Estimation
                    # --------------------------------------------------------------------------
                    CodeResidual = (CorrCode - GeomRange) if Dtr != Const.NAN else 0
                    PhaseResidual = (CorrPhase - GeomRange) if Dtr != Const.NAN else 0

                    # # Update Dictionary
                    SatCorrInfo["LeoX"] = RcvrRefPosXyz[0]
                    SatCorrInfo["LeoY"] = RcvrRefPosXyz[1]
                    SatCorrInfo["LeoZ"] = RcvrRefPosXyz[2]
                    
                    SatCorrInfo["LeoApoX"] = RcvrApoXyz[0]
                    SatCorrInfo["LeoApoY"] = RcvrApoXyz[1]
                    SatCorrInfo["LeoApoZ"] = RcvrApoXyz[2]
                    
                    SatCorrInfo["SatX"] = SatCopPos[0]
                    SatCorrInfo["SatY"] = SatCopPos[1]
                    SatCorrInfo["SatZ"] = SatCopPos[2]
                    
                    SatCorrInfo["SatApoX"] = Apo[0]
                    SatCorrInfo["SatApoY"] = Apo[1]
                    SatCorrInfo["SatApoZ"] = Apo[2]
                    
                    SatCorrInfo["FlightTime"] = FlightTime

                    SatCorrInfo["SatClk"] = SatClkBias
                    SatCorrInfo["SatCodeBia"] = satCodeBias
                    SatCorrInfo["SatPhaseBia"] = satPhaseBias

                    SatCorrInfo["Dtr"] = Dtr

                    SatCorrInfo["CorrCode"] = CorrCode 
                    SatCorrInfo["CorrPhase"] = CorrPhase 
                    SatCorrInfo["GeomRange"] = GeomRange
                    
                    SatCorrInfo["CodeResidual"] = CodeResidual
                    SatCorrInfo["PhaseResidual"] = PhaseResidual
                    SatCorrInfo["SigmaUere"] = UERE

                    # Update Flag = 0 [don't use] Set Flag to 0 if Dtr = Nan
                    if Dtr == Const.NAN:
                        SatCorrInfo["Flag"] = 0

                    # Put the Flag at 0 if the End of SatClkInfo file

                    # Accumulate Sigmas and residuals
                    if SatLabel[0] == "G":
                        ResidualsAccumGPS.append(CodeResidual)
                        UeresAccumGPS.append(UERE) 
                    elif SatLabel[0] == "E":
                        ResidualsAccumGAL.append(CodeResidual)
                        UeresAccumGAL.append(UERE)
                
                else:
                    SatCorrInfo["Flag"] = 0

            else:
                # Update Flag = 0 [don't use]
                SatCorrInfo["Flag"] = 0

        else:
            # Update Flag = 0 [don't use]
            SatCorrInfo["Flag"] = 0

        # Update Sod_1
        Sod_1[SatLabel].append(SatCorrInfo["Sod"])

        # Update SatComPos_1
        try:
            SatComPos_1[SatLabel].append([SatComPosSagnac[0], SatComPosSagnac[1], SatComPosSagnac[2]])
        except:
            SatComPos_1[SatLabel].append([0, 0, 0])

        # Update CorrInfo
        CorrInfo[SatLabel] = SatCorrInfo

        try:
            del SatComPosSagnac
        except:
            pass

    # Estimate the Receiver Clock guess as a weighted average of the Residuals
    #  but before we need to verify that there are no 0s in our arrays and not consider those measurements
    ResidualsAccumGPS = [res for i, res in enumerate(ResidualsAccumGPS) if UeresAccumGPS[i] != 0]
    UeresAccumGPS = [uere for uere in UeresAccumGPS if uere != 0]
    ResidualsAccumGAL = [res for i, res in enumerate(ResidualsAccumGAL) if UeresAccumGAL[i] != 0]
    UeresAccumGAL = [uere for uere in UeresAccumGAL if uere != 0]

    for SatLabel, SatCorrInfo in CorrInfo.items():
        # Filter the according to Flag status
        if SatCorrInfo["Flag"] !=0:
            # Compute the Receiver Clock
            if SatLabel[0] == "G":
                RcvrClk = estimateRcvrClk(ResidualsAccumGPS, UeresAccumGPS)
            elif SatLabel[0] == "E":  
                RcvrClk = estimateRcvrClk(ResidualsAccumGAL, UeresAccumGAL)
            #  Remove the Receiver Clock from the Residuals and update the fileds
            SatCorrInfo["CodeResidual"] = SatCorrInfo["CodeResidual"] - RcvrClk
            SatCorrInfo["PhaseResidual"] = SatCorrInfo["PhaseResidual"] - RcvrClk
            SatCorrInfo["RcvrClk"] = RcvrClk
    
    


    return CorrInfo, RcvrRefPosXyz, RcvrRefPosLlh
