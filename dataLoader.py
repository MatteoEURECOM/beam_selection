import pandas as pd
import numpy as np
ORIGINAL=False

### Loading Data ###
def loadDataTraining():
    filename = "coords_labels.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='train')
    X_pos = df[['X', 'Y', 'Z', 'LOS']].to_numpy()
    X_pos=X_pos[:,0:4]
    Y = df[['Labels']].to_numpy() #Channel gains
    Y=np.array(Y[:,0].tolist())
    Y = Y / Y.sum(axis=1)[:, None] #Normalize channel gains so that looks like probabilities
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-np.mean(X_pos[:,i])
        X_pos[:,i]=X_pos[:,i]/np.sqrt(np.mean(X_pos[:,i]**2))
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    if(ORIGINAL):
        X_lidar = lidar_to_2d(np.concatenate((np.load('./data/lidar_train.npz')['input'],np.load('./data/lidar_train2.npz')['input']),axis=0))
        X_lidar = np.expand_dims(X_lidar, axis=3)
    else:
        lidar = np.load('./lidar_008_2D.npz')
        X_lidar = lidar['arr_0']
        X_lidar = np.expand_dims(X_lidar, axis=3)
    return X_pos,(X_lidar+2)/3,Y

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
    if (ORIGINAL):
        X_lidar = lidar_to_2d(np.load('./data/lidar_val.npz')['input'])
        X_lidar = np.expand_dims(X_lidar, axis=3)
    else:
        lidar = np.load('./lidar_009_2D.npz')
        X_lidar = lidar['arr_0']
        X_lidar=np.expand_dims(X_lidar,axis=3)
    return X_pos,(X_lidar+2)/3,Y

def loadDataTesting():
    filename = "coords_labels_test.h5"  #Data frame with vehicles coordinates and LOS/NLOS indicator
    df = pd.read_hdf(filename, key='test')
    X_pos = df[['X', 'Y', 'Z']].to_numpy()
    X_pos=X_pos[:,0:3]
    labels = pd.read_csv('./beam_test_label_rowsfirst.csv', header=None)
    labels= labels.values.reshape(-1) # Channel gains
    Y = np.eye(256)[labels]
    #Normalizing  column-wise the dataframe, zero mean and unit variance
    m=[757.8467177554851,555.4583686176969,2.1526621404323745]
    c=[4.5697448219753305,68.02014004606737,0.9322994212410762]
    for i in range(0,3):
        X_pos[:,i]=X_pos[:,i]-m[i]
        X_pos[:,i]=X_pos[:,i]/c[i]
    #Normalizing lidar will cast all to float making it too big, normalize the input tensor instead
    if (ORIGINAL):
        X_lidar = lidar_to_2d(np.load('./data/lidar_test.npz')['input'])
        X_lidar = np.expand_dims(X_lidar, axis=3)
    else:
        lidar = np.load('./lidar_010_2D.npz')
        X_lidar = lidar['arr_0']
        X_lidar = np.expand_dims(X_lidar, axis=3)
    return X_pos,(X_lidar+2)/3,Y


def lidar_to_2d(lidar_data):


    lidar_data1 = np.zeros_like(lidar_data)[:, :, :, 1]

    lidar_data1[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_data1[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_data1[np.max(lidar_data == -1, axis=-1)] = -1

    return lidar_data1
