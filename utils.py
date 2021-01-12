import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD
import tensorflow as tf
import keras

FLATTENED=True
SUM=False
LIDAR_TYPE='ABSOLUTE'
'''
g = tf.Graph()
run_meta = tf.compat.v1.RunMetadata()
with g.as_default():
    vgg = multiModalNet()
    vgg.summary()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    if flops is not None:
        print('Flops should be ~',2*25*16*9)
        print('25 x 25 x 9 would be',2*25*25*9) # ignores internal dim, repeats first
        print('TF stats gives',flops.total_float_ops)
'''
def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops  # Prints the "flops" of the model.


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
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_original_labels.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE_LARGE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_large.npz',FLATTENED,SUM)
    if (Net == 'MULTIMODAL'):
        model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val,POS_val])  # Get predictions
    elif (Net == 'IPC'):
        model= LIDAR(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict(LIDAR_val)  # Get predictions
    elif(Net == 'GPS'):
        model = GPS()
        model.load_weights(saved_model)
        preds = model.predict(POS_val)  # Get predictions
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
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_original_labels.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE_LARGE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_large.npz',FLATTENED,SUM)
    NLOSind = np.where(NLOS_val == 0)[0]  # Get the NLoS users
    LOSind = np.where(NLOS_val== 1)[0]    # Get the LoS users
    if (Net == 'MULTIMODAL'):
        model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds_gains_NLOS = model.predict([LIDAR_val[NLOSind, :, :,:],POS_val[NLOSind,:]])  # Get predictions
        preds_gains_LOS = model.predict([LIDAR_val[LOSind, :, :, :],POS_val[LOSind, :]])  # Get predictions
    elif (Net == 'IPC'):
        model= LIDAR(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds_gains_NLOS = model.predict(LIDAR_val[NLOSind, :, :,:])  # Get predictions
        preds_gains_LOS = model.predict(LIDAR_val[LOSind, :, :,:])  # Get predictions
    elif (Net == 'GPS'):
        model= GPS()
        model.load_weights(saved_model)
        preds_gains_NLOS = model.predict(POS_val[NLOSind, :])  # Get predictions
        preds_gains_LOS = model.predict(POS_val[LOSind, :])  # Get predictions
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




def plotS010(preds_path):
    labels = pd.read_csv('./data/beam_test_label_columnsfirst.csv', header=None)
    labels = labels.values
    pred = pd.read_csv(preds_path, header=None)
    pred = pred.values
    curve_NLOS = np.zeros(256)
    for i in range(labels.shape[0]):
        curve_NLOS[np.where(pred[i, :] == labels[i])] += 1
    curve_NLOS = np.cumsum(curve_NLOS)
    return curve_NLOS / curve_NLOS[len(curve_NLOS) - 1]