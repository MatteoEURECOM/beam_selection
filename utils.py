import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import loadDataValidation
from models import multiModalNet,Lidar2D,GPS


def readHistory(path):
    history = pickle.load(open(path, "rb"))
    ### Plot Validation Acc and Validation Loss
    acc = history['categorical_accuracy']
    val_acc = history['val_categorical_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    t5 = history['val_top_5_accuracy']
    t10 = history['val_top_10_accuracy']
    t50 = history['val_top_50_accuracy']

    epochs = range(1, len(acc) + 1)
    plt.subplot(121)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc, 'b--', label='accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'g-', label='validation accuracy', linewidth=2)
    plt.legend()
    plt.subplot(122)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, 'b--', label='loss', linewidth=2)
    plt.plot(epochs, val_loss, 'g--', label='validation loss', linewidth=2)
    plt.legend()
    plt.savefig('TrainingCurves.png')
    plt.show()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, t5, 'r--', label='val_top_5_accuracy', linewidth=2)
    plt.plot(epochs, t10, 'b--', label='val_top_10_accuracy', linewidth=2)
    plt.plot(epochs, t50, 'k--', label='val_top_50_accuracy', linewidth=2)
    plt.legend()
    plt.savefig('TrainingCurves_1.png')
    plt.show()
    return

def plots009(saved_model,Net):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    X_p_val, X_l_val, Y_val = loadDataValidation()
    if (Net == 'MULTIMODAL'):
        model = multiModalNet()
        model.load_weights(saved_model)
        preds = model.predict([X_l_val[:, :, :,:],X_p_val[:, 0:3]])  # Get predictions
    elif (Net == 'IPC'):
        model = Lidar2D
        model.load_weights(saved_model)
        preds = model.predict(X_l_val[:, :, :,:])  # Get predictions
    elif(Net == 'GPS'):
        model = GPS
        model.load_weights(saved_model)
        preds = model.predict(X_p_val[:, 0:3])
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve

def plotNLOSvsLOS(saved_model,Net):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    X_p_val, X_l_val, Y_val = loadDataValidation()
    NLOSind = np.where(X_p_val[:, 3] == 0)[0]  # Get the NLoS users
    LOSind = np.where(X_p_val[:, 3] == 1)[0]    # Get the LoS users
    if (Net == 'MULTIMODAL'):
        model = multiModalNet()
        model.load_weights(saved_model)
        preds_gains_NLOS = model.predict([X_l_val[NLOSind, :, :,:],X_p_val[NLOSind, 0:3]])  # Get predictions
        preds_gains_LOS = model.predict([X_l_val[LOSind, :, :, :],X_p_val[LOSind, 0:3]])  # Get predictions
    elif (Net == 'IPC'):
        model = Lidar2D
        model.load_weights(saved_model)
        preds_gains_NLOS = model.predict(X_l_val[NLOSind, :, :,:])  # Get predictions
        preds_gains_LOS = model.predict(X_l_val[LOSind, :, :,:])  # Get predictions
    pred_NLOS= np.argsort(-preds_gains_NLOS, axis=1) #Descending order
    true_NLOS=np.argmax(Y_val[NLOSind,:], axis=1) #Best channel
    curve_NLOS=np.zeros(256)
    for i in range(0,len(pred_NLOS)):
        curve_NLOS[np.where(pred_NLOS[i,:] == true_NLOS[i])]=curve_NLOS[np.where(pred_NLOS[i,:] == true_NLOS[i])]+1
    curve_NLOS=np.cumsum(curve_NLOS)
    pred_LOS= np.argsort(-preds_gains_LOS, axis=1) #Descending order
    true_LOS=np.argmax(Y_val[LOSind,:], axis=1) #Best channel
    curve_LOS=np.zeros(256)
    for i in range(0,len(pred_LOS)):
        curve_LOS[np.where(pred_LOS[i,:] == true_LOS[i])]=curve_LOS[np.where(pred_LOS[i,:] == true_LOS[i])]+1
    curve_LOS=np.cumsum(curve_LOS)
    return curve_LOS,curve_NLOS



def reorder(data, num_rows, num_columns):
    '''
    Reorder a vector obtained from a matrix: read row-wise and write column-wise.
    '''
    original_vector  = np.asarray(data, dtype = np.float)
    #read row-wise
    original_matrix = np.reshape(original_vector, (num_rows, num_columns))
    #write column-wise
    new_vector = np.reshape(original_matrix, num_rows*num_columns, 'F')
    return new_vector

def plotS010(preds_path):
    labels = pd.read_csv('./beam_test_label_columnsfirst.csv', header=None)
    labels = labels.values
    pred = pd.read_csv(preds_path, header=None)
    pred = pred.values
    curve_NLOS = np.zeros(256)
    for i in range(labels.shape[0]):
        curve_NLOS[np.where(pred[i, :] == labels[i])] += 1
    curve_NLOS = np.cumsum(curve_NLOS)
    return curve_NLOS / curve_NLOS[len(curve_NLOS) - 1]