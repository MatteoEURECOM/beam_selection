import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#S008 mean and standard deviation to normalize GPS inputs
s008_mean=[757.8,555.4,2.1]
s008_SD=[4.5,68.0,0.9]

def lidar_to_2d(lidar_data):
    lidar_data1 = np.zeros_like(lidar_data)[:, :, :, 1]
    lidar_data1[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_data1[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_data1[np.max(lidar_data == -1, axis=-1)] = -1
    return lidar_data1

def load_dataset(filename,FLATTENED):
    npzfile = np.load(filename)
    POS=npzfile['POS']
    for i in range(0,3):
        POS[:,i]=(POS[:,i]-s008_mean[i])/s008_SD[i]
    LIDAR=npzfile['LIDAR']
    if(FLATTENED):
        LIDAR = np.expand_dims(lidar_to_2d(LIDAR), axis=3)
    Y=npzfile['Y']
    if '010' in filename:
        LOS=0
    else:
        LOS=npzfile['LOS']
    return POS,((LIDAR+2.0)/3.0).astype(np.float32),Y,LOS

