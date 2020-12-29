import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import sys
import os
import csv
import shutil
import cv2




def loadDataValidation():
    filename = "coords_labels.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='val')
    X_pos = df[['X', 'Y', 'Z', 'LOS']].to_numpy()
    X_pos=X_pos[:,0:4]
    Y = df[['Labels']].to_numpy() #Channel gains
    Y=np.array(Y[:,0].tolist())
    Y = Y / Y.sum(axis=1)[:, None] #Normalize channel gains so that looks like probabilities
    m=[757.8467177554851,555.4583686176969,2.1526621404323745]  #Training set means
    c=[4.5697448219753305,68.02014004606737,0.9322994212410762] #Training set variances
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-m[i]
        X_pos[:,i]=X_pos[:,i]/c[i]
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    lidar = np.load('./lidar_010.npz')
    X_lidar = lidar['input']
    return X_pos,X_lidar,Y


lidar = np.load('./lidar_010_3D.npz')
X_lidar = lidar['arr_0']


j=0
DIM=2

X_p_val,X_l_val,Y_val=loadDataValidation()
Out2D=np.zeros((X_l_val.shape[0],20,200))
for i in range(X_l_val.shape[0]):
    test=X_l_val[i,75:115,30:,:]
    div=np.count_nonzero(test, axis=2)
    tx,ty,_=np.where(test==-1)
    rx,ry, _ = np.where(test == -2)
    if(tx.size==0):
        j+=1
    else:
        if(DIM==2):
            sum = np.sum(test,axis=2)
            sum[tx,ty]=0
            sum[rx,ry]=0
            average=np.round(np.divide(sum,div+0.0001))
            Out2D[i,:,:] = cv2.resize(average, dsize=(200, 20), interpolation=cv2.INTER_AREA)
            Out2D[i,np.int(tx/2),np.int(ty/1.5)]=-1
            Out2D[i, np.int(rx/2), np.int(ry / 1.5)] = -2
            plt.imshow(Out2D[i, :, :])
            plt.colorbar()
            plt.show()
            plt.imshow(sum)
            plt.colorbar()
            plt.show()

        else:
            sum = np.sum(test, axis=2)
            sum[tx, ty] = 0
            sum[rx, ry] = 0
            average = np.round(np.divide(sum, 10))
            Out2D[i, :, :] = cv2.resize(average, dsize=(200, 20), interpolation=cv2.INTER_NEAREST)
            Out2D[i, np.int(tx / 2), np.int(ty / 1.5)] = -1
            Out2D[i, np.int(rx / 2), np.int(ry / 1.5)] = -2
            plt.imshow(Out2D[i, :, :])
            plt.colorbar()
            plt.show()
            plt.imshow(sum)
            plt.colorbar()
            plt.show()

print(np.min( Out2D[10,:,:]))
np.savez('./lidar_010_3D.npz',Out2D)


lidar = np.load('./lidar_010_3D.npz')
X_lidar = lidar['arr_0']
print(np.min(X_lidar[10,:,:]))