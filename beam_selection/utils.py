import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD,MIXTURE,NON_LOCAL_MIXTURE
import tensorflow as tf
import keras

FLATTENED=True
SUM=False

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


def plotNLOSvsLOS(saved_model,Net,LIDAR_TYPE='ABSOLUTE'):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_original_labels.npz',FLATTENED,SUM)
        POS_val=POS_val[:,0:2]
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
    elif (Net == 'MIXTURE'):
        model = MIXTURE(FLATTENED, LIDAR_TYPE)
        model.load_weights(saved_model)
        preds_gains_NLOS =  model.predict([LIDAR_val[NLOSind, :,:,:]* 3 - 2,POS_val[NLOSind, :]])  # Get predictions
        preds_gains_LOS =  model.predict([LIDAR_val[LOSind, :,:,:]* 3 - 2,POS_val[LOSind, :]])  # Get predictions
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

def plots009(saved_model,Net,LIDAR_TYPE='ABSOLUTE',EMBEEDING='embedded',intermediate_dim=1):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_original_labels.npz',FLATTENED,SUM)
        POS_val=POS_val[:,0:2]
    elif LIDAR_TYPE=='ABSOLUTE_LARGE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_large.npz',FLATTENED,SUM)
    if (Net == 'MULTIMODAL'):
        model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val,POS_val])  # Get predictions
    elif (Net == 'MULTIMODAL_OLD'):
        model= MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds =  model.predict([LIDAR_val* 3 - 2,POS_val])  # Get predictions
    elif (Net == 'IPC'):
        model= LIDAR(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict(LIDAR_val)  # Get predictions
    elif(Net == 'GPS'):
        model = GPS()
        model.load_weights(saved_model)
        preds = model.predict(POS_val)  # Get predictions
    elif(Net == 'MIXTURE'):
        model = MIXTURE(FLATTENED, LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val* 3 - 2,POS_val])    # Get predictions
    elif(Net == 'NON_LOCAL_MIXTURE'):
        model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE,EMBEEDING,intermediate_dim)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val* 3 - 2,POS_val])    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve


def plots010(saved_model,Net,LIDAR_TYPE='ABSOLUTE',EMBEEDING='embedded',intermediate_dim=1):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s010_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s010_original_labels.npz',FLATTENED,SUM)
        POS_val=POS_val[:,0:2]
    elif LIDAR_TYPE=='ABSOLUTE_LARGE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s010_large.npz',FLATTENED,SUM)
    if (Net == 'MULTIMODAL'):
        model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val,POS_val])  # Get predictions
    elif (Net == 'MULTIMODAL_OLD'):
        model= MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds =  model.predict([LIDAR_val* 3 - 2,POS_val]) # Get predictions
    elif (Net == 'IPC'):
        model= LIDAR(FLATTENED,LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict(LIDAR_val)  # Get predictions
    elif(Net == 'GPS'):
        model = GPS()
        model.load_weights(saved_model)
        preds = model.predict(POS_val)  # Get predictions
    elif(Net == 'MIXTURE'):
        model = MIXTURE(FLATTENED, LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val* 3 - 2,POS_val])    # Get predictions
    elif(Net == 'NON_LOCAL_MIXTURE'):
        model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE,EMBEEDING,intermediate_dim)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val* 3 - 2,POS_val])    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve

def readHistoryMC(path,reps):

    history = pickle.load(open(path, "rb"))
    ### Plot Validation Acc and Validation Loss
    acc = history['categorical_accuracy']
    max_epoch=int((len(acc) + 1)/reps)
    epochs = range(0, max_epoch)
    acc=np.reshape(acc,(-1,max_epoch))
    val_acc = history['val_categorical_accuracy']
    val_acc=np.reshape(val_acc,(-1,max_epoch))
    loss = history['loss']
    loss=np.reshape(loss,(-1,max_epoch))
    val_loss = history['val_loss']
    val_loss=np.reshape(val_loss,(-1,max_epoch))
    t5 = history['val_top_5_accuracy']
    t5=np.reshape(t5,(-1,max_epoch))
    t10 = history['val_top_10_accuracy']
    t10=np.reshape(t10,(-1,max_epoch))
    t50 = history['val_top_50_accuracy']
    t50=np.reshape(t50,(-1,max_epoch))




    plt.title('Anti-curriculum Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, np.mean(acc,axis=0), 'b--', label='Accuracy', linewidth=2)
    std=np.sqrt(np.mean((acc-np.mean(acc,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(acc,axis=0)-std, np.mean(acc,axis=0)+std,color="lightsteelblue")
    plt.plot(epochs, np.mean(val_acc,axis=0), 'g-', label='Validation Accuracy', linewidth=2)
    std=np.sqrt(np.mean((val_acc-np.mean(val_acc,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(val_acc,axis=0)-std, np.mean(val_acc,axis=0)+std,color="mediumseagreen")
    plt.legend()
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0.15,0.75])
    plt.savefig('Accuracy_ANTI.png', dpi=150)
    plt.show()

    plt.title('Anti-curriculum Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, np.mean(loss,axis=0), 'b--', label='Loss', linewidth=2)
    std=np.sqrt(np.mean((loss-np.mean(loss,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(loss,axis=0)-std, np.mean(loss,axis=0)+std,color="lightsteelblue")
    plt.plot(epochs, np.mean(val_loss,axis=0), 'g--', label='Validation Loss', linewidth=2)
    std=np.sqrt(np.mean((val_loss-np.mean(val_loss,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(val_loss,axis=0)-std, np.mean(val_loss,axis=0)+std,color="mediumseagreen")
    plt.legend()
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0.6,1.4])
    plt.savefig('Loss_ANTI.png', dpi=150)
    plt.show()

    plt.title('Anti-curriculum Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, np.mean(t5,axis=0), 'r--', label='Top 5', linewidth=2)
    std=np.sqrt(np.mean((t5-np.mean(t5,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(t5,axis=0)-std, np.mean(t5,axis=0)+std,color="salmon")
    plt.plot(epochs, np.mean(t10,axis=0), 'b--', label='Top 10', linewidth=2)
    std=np.sqrt(np.mean((t10-np.mean(t10,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(t10,axis=0)-std, np.mean(t10,axis=0)+std,color="lightsteelblue")
    plt.plot(epochs, np.mean(t50,axis=0), 'g--', label='Top 50', linewidth=2)
    std=np.sqrt(np.mean((t50-np.mean(t50,axis=0))**2,axis=0))
    plt.fill_between(epochs, np.mean(t50,axis=0)-std, np.mean(t50,axis=0)+std,color="mediumseagreen")
    plt.legend()
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0.65,1])
    plt.savefig('Metrics_ANTI.png', dpi=150)
    plt.show()
    return


def finalMCAvg(path,reps):

    history = pickle.load(open(path, "rb"))
    ### Plot Validation Acc and Validation Loss
    acc = history['categorical_accuracy']
    max_epoch=int((len(acc) + 1)/reps)
    epochs = range(0, max_epoch)
    acc=np.reshape(acc,(-1,max_epoch))
    val_acc = history['val_categorical_accuracy']
    val_acc=np.reshape(val_acc,(-1,max_epoch))
    loss = history['loss']
    loss=np.reshape(loss,(-1,max_epoch))
    val_loss = history['val_loss']
    val_loss=np.reshape(val_loss,(-1,max_epoch))
    t5 = history['val_top_5_accuracy']
    t5=np.reshape(t5,(-1,max_epoch))
    t10 = history['val_top_10_accuracy']
    t10=np.reshape(t10,(-1,max_epoch))
    t50 = history['val_top_50_accuracy']
    t50=np.reshape(t50,(-1,max_epoch))

    val_acc_mean=np.mean(val_acc,axis=0)[-1]
    val_acc_std=np.sqrt(np.mean((val_acc-np.mean(val_acc,axis=0))**2,axis=0))[-1]

    t5_mean=np.mean(t5,axis=0)[-1]
    t5_std=np.sqrt(np.mean((t5-np.mean(t5,axis=0))**2,axis=0))[-1]

    t10_mean=np.mean(t10,axis=0)[-1]
    t10_std=np.sqrt(np.mean((t10-np.mean(t10,axis=0))**2,axis=0))[-1]

    return val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std


