from InputOutput import LeoPosIdx, LeoQuatIdx, SatPosIdx, SatApoIdx, SatClkIdx, SatBiaIdx 
from collections import OrderedDict
from COMMON.Dates import convertYearDoy2JulianDay
import pandas as pd
import numpy as np
from GnssConstants import S_IN_D, KM_TO_M, SPEED_OF_LIGHT, OMEGA_EARTH, S_TO_MS, MM_TO_M, NS_TO_S, NAN
from math import pi
from COMMON.Misc import crossProd

def computeLeoComPos(LeoPosInfo, KM_TO_M):
    RcvrRefPosXyzCom = OrderedDict({})
    RcvrRefPosXyzCom = pd.DataFrame(RcvrRefPosXyzCom)

    RcvrRefPosXyzCom["SOD"] = LeoPosInfo[LeoPosIdx["SOD"]]

    RcvrRefPosXyzCom["xCM"] = LeoPosInfo[LeoPosIdx["xCM"]] * KM_TO_M
    RcvrRefPosXyzCom["yCM"] = LeoPosInfo[LeoPosIdx["yCM"]] * KM_TO_M
    RcvrRefPosXyzCom["zCM"] = LeoPosInfo[LeoPosIdx["zCM"]] * KM_TO_M

    return RcvrRefPosXyzCom


def computeSatClkBias(current_epoch, SatLabel, SatClkInfo):
        '''
            Goal:
                Compute Satellite Clock Bias by interpolation, by interpolatuing over the CLK file

            Inputs:
                current_epoch = current Sod
                SatLabel = Satellite label containing PRN and Constellation 
                SatClkInfo = dataframe copntaining CLK biases for all PRNs
            
            Returns :
                SatClkBias
            
        '''
        # Compute interpolation for Clk Bias
        SatClkBias = np.interp(current_epoch,
                                 SatClkInfo[(SatClkInfo[SatClkIdx["CONST"]] == SatLabel[0]) & (SatClkInfo[SatClkIdx["PRN"]] == int(SatLabel[1:]))][SatClkIdx["SOD"]],
                                 SatClkInfo[(SatClkInfo[SatClkIdx["CONST"]] == SatLabel[0]) & (SatClkInfo[SatClkIdx["PRN"]] == int(SatLabel[1:]))][SatClkIdx["CLK-BIAS"]]
                                 )
        
        maxSodSatCLKInfo = SatClkInfo[SatClkIdx["SOD"]]

        
        return SatClkBias, maxSodSatCLKInfo


def computeRcvrApo(Conf, Year, Doy, Sod, SatLabel, LeoQuatInfo):
    '''
            Goal:
                Compute Satellite Clock Bias by interpolation

            Inputs:
                PRN_Sod_1 = Sod_1[SatLabel] --> Previous Sod
                current_Sod = SatCorrInfo["Sod"] --> Current Sod
                SatClkInfo
            
            Returns :
                SatClkBias
            
        '''
    
    # 1. Configured CoM, Antenna Reference Point (ARP) position and PCOs are referred to the Satellite Reference Frame (SRF)
    confLeoXCoM, confLeoYCoM, confLeoZCoM = Conf["LEO_COM_POS"]
    confLeoXARP, confLeoYARP, confLeoZARP = Conf["LEO_ARP_POS"]
    
    # 2. The PCOs are the offset between the ARP and the APC
    if SatLabel[0] == "G":
        confLeoXPco, confLeoYPco, confLeoZPco = Conf["LEO_PCO_GPS"]
    elif SatLabel[0] == "E":
        confLeoXPco, confLeoYPco, confLeoZPco = Conf["LEO_PCO_GAL"]

    #   So the Antenna phase offset of the LEO is APO_Leo = (ARP_Leo + PCO_Leo) - CoM_Leo
    APO_SFR = np.array([(confLeoXARP + confLeoXPco) - confLeoXCoM, 
                        (confLeoYARP + confLeoYPco) - confLeoYCoM, 
                        (confLeoZARP + confLeoZPco) - confLeoZCoM])

    # 3. Apply the Satellite Quaternions to rotate the SFR towards the Earth Centered Inertial (ECI) Reference Frame, 
    #   Extract the quaternions for the given Sod and build the Rotation Matrix
    q0 = LeoQuatInfo[LeoQuatInfo[LeoQuatIdx["SOD"]] == Sod][LeoQuatIdx["q0"]].iloc[0]
    q1 = LeoQuatInfo[LeoQuatInfo[LeoQuatIdx["SOD"]] == Sod][LeoQuatIdx["q1"]].iloc[0]
    q2 = LeoQuatInfo[LeoQuatInfo[LeoQuatIdx["SOD"]] == Sod][LeoQuatIdx["q2"]].iloc[0]
    q3 = LeoQuatInfo[LeoQuatInfo[LeoQuatIdx["SOD"]] == Sod][LeoQuatIdx["q3"]].iloc[0]
    
    R = np.array([[1-2*(q2**2)-2*(q3**2), 2*(q1*q2-q0*q3),       2*(q0*q2+q1*q3)],
                  [2*(q1*q2+q0*q3),       1-2*(q1**2)-2*(q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2),       2*(q0*q1+q2*q3),       1-2*(q1**2)-2*(q2**2)]
                  ])
    
    APO_ECI = R @ APO_SFR

    # 4. Convert ECI Coordinates to ECEF coordinates with the simplified model for the Greenwich Sideral Time
    JDN = convertYearDoy2JulianDay(Year, Doy, Sod) - 2415020
    fday = Sod / S_IN_D
    gstr = ((279.690983 + 0.9856473354 * JDN + 360 * fday + 180) % 360) * pi/180

    Rgstr = np.array([[np.cos(gstr),  np.sin(gstr), 0],
                      [-np.sin(gstr), np.cos(gstr), 0],
                      [0,             0,            1]])
    
    # Calculate APO
    APO_ECEF = Rgstr @ APO_ECI
    RcvrApoXyz = APO_ECEF
    
    return RcvrApoXyz


def computeSatComPos(TransmissionTime, SatLabel, SatPosInfo):
    '''
        Goal:
            Compute Satellite Center of Masses (CoM), by using Lagrangian interpolation.

        Inputs:
            TransmissionTime = Transmission time of that specific PRN, based on C1.
            SatLabel = PRN identifier.
            SatPosInfo = dataframe containing the satellite position every 300s (based on SP3 file).

        Returns:
            SatComPos = Interpolated satellite position at TransmissionTime (in meters).
    '''
    num_of_points = 10  # We need 5 before and 5 after the TransmissionTime
    half_points = num_of_points // 2  # 5 points before and 5 points after

    # 1. Filter the SatPosInfo dictionary based on the Constellation and the PRN
    satData = SatPosInfo[(SatPosInfo[SatPosIdx["CONST"]] == SatLabel[0]) &
                         (SatPosInfo[SatPosIdx["PRN"]] == int(SatLabel[1:]))].reset_index(drop=True)
    
    # 2. Identify the interval where the TransmissionTime fits
    try:
        # Index of the last point that is <= TransmissionTime
        interval_starts_idx = satData[satData[SatPosIdx["SOD"]] <= TransmissionTime].index[-1]
        # Index of the first point that is > TransmissionTime
        interval_ends_idx = satData[satData[SatPosIdx["SOD"]] > TransmissionTime].index[0]
    except IndexError:
        raise ValueError("TransmissionTime is out of range in the satellite data.")

    # 3. Check if we're near the start of the file
    if interval_starts_idx < half_points:
        # Not enough points before TransmissionTime, take more from after
        start_idx = 0  # Start from the first point
        end_idx = num_of_points  # Take the first 10 points
    # 4. Check if we're near the end of the file
    elif interval_ends_idx + half_points > len(satData):
        # Not enough points after TransmissionTime, take more from before
        end_idx = len(satData)  # Go until the last point
        start_idx = len(satData) - num_of_points  # Go back to get 10 points total
    # 5. General case, enough points before and after TransmissionTime
    else:
        start_idx = interval_starts_idx - half_points  # 5 points before
        end_idx = interval_starts_idx + half_points + 1  # 5 points after

    # 6. Extract the time points and the satellite positions (X, Y, Z) for interpolation
    t_points = satData[SatPosIdx["SOD"]][start_idx:end_idx]

    # Extract the XYZ satellite positions
    satPosXYZ = np.array([satData[SatPosIdx["xCM"]][start_idx:end_idx].values * KM_TO_M,
                          satData[SatPosIdx["yCM"]][start_idx:end_idx].values * KM_TO_M,
                          satData[SatPosIdx["zCM"]][start_idx:end_idx].values * KM_TO_M])

    # 7. Perform Lagrange interpolation using the time points and the XYZ positions
    Lx, Ly, Lz = LangrangeInterpolation(TransmissionTime, t_points, satPosXYZ)

    # 8. Return the interpolated satellite position (X, Y, Z)
    SatComPos = np.array([Lx, Ly, Lz])

    return SatComPos



def LangrangeInterpolation(TransmissionTime, t_points, satPosXYZ):
    '''
        Goal: 
            Perform Lagrange interpolation for 3D positions based on time.
        
        Inputs:
            t_points (list or array-like): Time points for Lagrange interpolation.
            TransmissionTime (float): The specific time at which interpolation is performed.
            satPosXYZ (array-like): XYZ positions corresponding to the time points.
        
        Returns:
            tuple: Interpolated Lagrange polynomials for X, Y, Z positions (Lx, Ly, Lz).
    '''
    
    # Ensure inputs are compatible (convert to NumPy arrays if necessary)
    t_points = np.array(t_points)
    satPosXYZ = np.array(satPosXYZ)
    
    # Ensure the dimensions match
    assert len(t_points) == satPosXYZ.shape[1], "The number of time points must match the number of position points."
    
    # Initialize Lagrange interpolating polynomials for X, Y, Z
    Lx, Ly, Lz = 0.0, 0.0, 0.0
    
    # Iterate through all points
    for i in range(len(t_points)):
        # Calculate the Lagrange basis polynomial L_i(x)
        L_i = 1.0
        for j in range(len(t_points)):
            if i != j:
                # Compute the Lagrange basis polynomial term
                L_i *= (TransmissionTime - t_points[j]) / (t_points[i] - t_points[j])
        
        # Add the interpolated values for X, Y, and Z
        Lx += satPosXYZ[0][i] * L_i
        Ly += satPosXYZ[1][i] * L_i
        Lz += satPosXYZ[2][i] * L_i
    
    # Return the interpolated position for X, Y, and Z
    return Lx, Ly, Lz



def computeFlightTime(SatComPos, RcvrRefPosXyz):
    '''
        Goal: 
            To compute the flight time of the satellite
        
        Inputs:
            SatComPos = Satellite Center of masses positions
            RcvrRefPosXyz = Receiver reference positon corrected by APO
        
        Returns:
            FlightTime
    '''
    x = SatComPos[0] - RcvrRefPosXyz[0]
    y = SatComPos[1] - RcvrRefPosXyz[1]
    z = SatComPos[2] - RcvrRefPosXyz[2]

    norm = np.sqrt(x**2 + y**2 + z**2)

    FlightTime = (norm / SPEED_OF_LIGHT)

    return FlightTime


def computeSagnacCorr(SatComPos, FlightTime):
    '''
        Goal: 
            To compute the Sagnac correction
        
        Inputs:
            SatComPos = Satellite Center of masses positions
            FlightTime = Satellite flight time
        
        Returns:
            SatComPosSagnac
    '''
    theta = OMEGA_EARTH * FlightTime/S_TO_MS

    R3 = np.array([[np.cos(theta),  np.sin(theta),  0],
                   [-np.sin(theta), np.cos(theta),  0],
                   [0,              0,              1]])

    SatComPosSagnac = R3 @ SatComPos 
    
    return SatComPosSagnac


def computeSatApo(SatComPosSagnac, SunPos, GammaF1F2, SatApoInfo):
    '''
        Goal: 
            To compute the Satellite APO and satellite rAPC
        
        Inputs:
            SatComPosSagnac = Satellite Center of masses positions corrected
            SunPos = Position of the sun
            GammaF1F2 = Gamma coeficiente between L1L2 or E1E5 depending on GPS or Galileo Constellation
            SatApoInfo = dictionary continaing the Satellite APO information
        
        Returns:
            APO
            rAPO
    '''
    # Compute the CoM position relative to the Sun, and its norm
    rSunComDiff = SunPos - SatComPosSagnac

    # Compute unitary vectors
    k = - SatComPosSagnac / np.linalg.norm(SatComPosSagnac)
    e = rSunComDiff / np.linalg.norm(rSunComDiff)
    
    # Compute the Norm of the j vector
    jNonUnitary = crossProd(k, e)
    j = jNonUnitary / np.linalg.norm(jNonUnitary)

    # Compute i
    i = crossProd(j, k)
    
    # Build the Rotation matrix
    R = np.column_stack((i, j, k))

    # Build IF measurements for the APOs
    SatAPOL1 = np.array([SatApoInfo[SatApoIdx["x_f1"]].iloc[0], 
                      SatApoInfo[SatApoIdx["y_f1"]].iloc[0], 
                      SatApoInfo[SatApoIdx["z_f1"]].iloc[0]])
     
    SatAPOL2 = np.array([SatApoInfo[SatApoIdx["x_f2"]].iloc[0], 
                     SatApoInfo[SatApoIdx["y_f2"]].iloc[0], 
                     SatApoInfo[SatApoIdx["z_f2"]].iloc[0]]) 
    
    SatAPO = ((SatAPOL2 - (GammaF1F2 * SatAPOL1))/ (1 - GammaF1F2)) / MM_TO_M

    # Get the rAPC
    satAPORotated = np.dot(R, SatAPO)

    return satAPORotated


def getBiases(GammaF1F2, SatBiaInfo):
    '''
        Goal: 
            To compute the Satellite Biases
        
        Inputs:
            GammaF1F2 = Gamma coeficiente between L1L2 or E1E5 depending on GPS or Galileo Constellation
            SatBiaInfo = dictionary continaing the Satellite Biases information
        
        Returns:
            satCodeBias
            SatPhaseBias
    '''
    # Code Biases
    SatCodeClkBiasL1 = SatBiaInfo[SatBiaIdx["CLK_f1_C"]].iloc[0]
    SatCodeClkBiasL2 = SatBiaInfo[SatBiaIdx["CLK_f2_C"]].iloc[0]
    SatCodeObsBiasL1 = SatBiaInfo[SatBiaIdx["OBS_f1_C"]].iloc[0]
    SatCodeObsBiasL2 = SatBiaInfo[SatBiaIdx["OBS_f2_C"]].iloc[0]

    # Phase Biases
    SatPhaseClkBiasL1 = SatBiaInfo[SatBiaIdx["CLK_f1_P"]].iloc[0]
    SatPhaseClkBiasL2 = SatBiaInfo[SatBiaIdx["CLK_f2_P"]].iloc[0]
    SatPhaseObsBiasL1 = SatBiaInfo[SatBiaIdx["OBS_f1_P"]].iloc[0]
    SatPhaseObsBiasL2 = SatBiaInfo[SatBiaIdx["OBS_f2_P"]].iloc[0]

    # Compute IFBs and Code and Pahase IF obs
    satIfbCode = ((SatCodeClkBiasL2 - (GammaF1F2 * SatCodeClkBiasL1))/ (1 - GammaF1F2)) 
    satIfbPhase = ((SatPhaseClkBiasL2 - (GammaF1F2 * SatPhaseClkBiasL1))/ (1 - GammaF1F2)) 
    
    satCodeBiasIF = ((SatCodeObsBiasL2 - (GammaF1F2 * SatCodeObsBiasL1))/ (1 - GammaF1F2)) 
    satPhaseBiasIF = ((SatPhaseObsBiasL2 - (GammaF1F2 * SatPhaseObsBiasL1))/ (1 - GammaF1F2)) 

    # Compute Sat Code Bias without Clk
    satCodeBias = (satIfbCode - satCodeBiasIF) * NS_TO_S * SPEED_OF_LIGHT
    satPhaseBias = (satIfbPhase - satPhaseBiasIF) * NS_TO_S * SPEED_OF_LIGHT
    # satCodeBias = satCodeBiasIF * NS_TO_S * SPEED_OF_LIGHT
    # satPhaseBias = satPahseBiasIF * NS_TO_S * SPEED_OF_LIGHT

    return  satCodeBias, satPhaseBias


def computeDtr(SatComPos_1, SatComPos, Sod, Sod_1):
    '''
        Goal: 
            To compute the DTR
        
        Inputs:
            SatComPosSagnac_1 = Satellite COM corrected from previous epoch
            SatComPosSagnac = Satellite COM corrected from current epoch
            Sod = Satellite SOD from previous epoch
            Sod_1 = Satellite SOD from current epoch
        
        Returns:
            Dtr
    '''
    
    if np.array_equal(SatComPos_1, np.zeros(3)):
        Dtr = NAN
        
    else:
        # Determine Satellite positions from one epoch to the next one
        rComDiff = SatComPos - SatComPos_1
        sodDiff = Sod - Sod_1
        # Calulate speed
        v = rComDiff / sodDiff
        # Compute DTR
        Dtr = -2 * np.dot(SatComPos, v) / (SPEED_OF_LIGHT)
    
    return Dtr
    

def getUere(Conf, SatLabel):
    '''
        Goal: 
            To extract the sigma UERE from the Configuration
        
        Inputs:
            Conf = Configuration data
            SatLabel = Satellite PRN and constellation
        
        Returns:
            UERE
    '''

    if SatLabel[0] == "G":
        UERE = Conf["GPS_UERE"]
    elif SatLabel[0] == "E":
        UERE = Conf["GAL_UERE"]

    return UERE


def computeGeomRange(SatCopPos, RcvrRefPosXyz):
    '''
        Goal:
            To compute the Geometrical range between the satellite and the receiver
        
        Inputs:
            SatCopPos = Satellite Center of Phases Position
            RcvrRefPosXyz = Receiver reference position with Antenna phase offsets Applied
        
        Returns:
            GeomRange
    '''

    GeomRange = np.linalg.norm(SatCopPos - RcvrRefPosXyz) 

    return GeomRange


def estimateRcvrClk(ResidualsAccum, UeresAccum):
    '''
        Goal:
            To compute the wieghted least squares Clk for the receiver
        
        Inputs:
            CodeResidual = Code residuals by subtracting the Geometrical Range
            UERE = Sigma Uere obtained from File

        Returns:
            RcvrClk
    
    '''
    # Compute the weights of the sigmas
    ResidualsAccum
    UeresAccum
    weights = [1/(uere**2) for uere in UeresAccum] 
    # weightsSum = np.sum(weights)

    X = np.ones((len(ResidualsAccum), 1))

    W = np.diag(weights)

    # Design matrix (in this case, a column vector of ones for estimating a constant)
    X = np.ones((len(ResidualsAccum), 1))

    # Weight matrix (diagonal matrix of weights)
    W = np.diag(weights)

    # Compute the WLS estimate using matrix formula
    X_T_W_X_inv = np.linalg.inv(X.T @ W @ X)
    X_T_W_y = X.T @ W @ ResidualsAccum
    RcvrClk = X_T_W_X_inv @ X_T_W_y
    
    # Formulate the Weighted Least Square solution
    # weightedResiduals = np.sum(np.array(weights) * np.array(ResidualsAccum))
    
    # Compute the clock by dividing the weighted residuals by the sum of the weights

    return RcvrClk