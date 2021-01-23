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
LIDAR_TYPE='ABSOLUTE'

g = tf.Graph()
run_meta = tf.compat.v1.RunMetadata()
with g.as_default():
    net = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE)
    net.summary()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    if flops is not None:
        print('TF stats gives',flops.total_float_ops)

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
        model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE)
        model.load_weights(saved_model)
        preds = model.predict([LIDAR_val* 3 - 2,POS_val])    # Get predictions
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
    elif (Net == 'MIXTURE'):
        model = MIXTURE(FLATTENED, LIDAR_TYPE)
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

def plots010(saved_model,Net):
    ### Plot Validation Accuracy for LoS and NLoS channels
    #NLOS
    if LIDAR_TYPE=='CENTERED':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s010_centered.npz',FLATTENED,SUM)
    elif LIDAR_TYPE=='ABSOLUTE':
        POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s010_original_labels.npz',FLATTENED,SUM)
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
        model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE)
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




val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMULTIMODAL_OLD_BETA_8_CURR',5)
plt.errorbar([1.2,2.2,3.2], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label='Multimodal Old',linestyle='None', marker='.',capsize=2,color="salmon")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryNON_LOCAL_MIXTURE_BETA_8_CURR',5)
plt.errorbar([1,2,3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label='Non Local Mixture',linestyle='None', marker='.',capsize=2,color="lightsteelblue")
'''
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMIXTURE_BETA_8_ONLY_LOS',5)
plt.errorbar([1,2,3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label='Only LOS',linestyle='None', marker='.',capsize=2,color="darkgray")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMIXTURE_BETA_8_ONLY_NLOS',5)
plt.errorbar([1.1,2.1,3.1], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label='Only NLOS',linestyle='None', marker='.',capsize=2,color="khaki")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMIXTURE_BETA_8_ANTI',5)
plt.errorbar([1.2,2.2,3.2], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label='Anti-curriculum',linestyle='None', marker='.',capsize=2,color="salmon")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMIXTURE_BETA_8_VANILLA',5)
plt.errorbar([1.3,2.3,3.3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std],  label='Vanilla',linestyle='None', marker='.',capsize=2,color="lightsteelblue")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('saved_models/HistoryMIXTURE_BETA_8_CURR',5)
'''
plt.errorbar([1.4,2.4,3.4], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std],label='Curriculum', linestyle='None', marker='.',capsize=2,color="mediumseagreen")
axes = plt.gca()
axes.set_yticks(np.arange(0.3, 1, 0.05))
axes.set_yticks(np.arange(0.3, 1, 0.01), minor=True)
axes.set_xticks([1.2,2.2,3.2])
axes.set_xticklabels(['Top-1','Top-5','Top-10'])
axes.grid(which='minor', alpha=0.2)
axes.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig('METRICS_CURR_NON_LOCAL.png', dpi=150)
plt.clf()

curve = plots009('saved_models/MULTIMODAL_OLD_BETA_8_CURR.h5', 'MULTIMODAL_OLD')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Multimodal Old',linestyle='--',color="salmon")

curve = plots009('saved_models/NON_LOCAL_MIXTURE_BETA_8_CURR.h5', 'NON_LOCAL_MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Non Local Mixture',linestyle='--',color="lightsteelblue")
'''
curve = plots009('saved_models/MIXTURE_BETA_8_ONLY_LOS.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Only LOS',linestyle='--',color="darkgray")
curve = plots009('saved_models/MIXTURE_BETA_8_ONLY_NLOS.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Only NLOS',linestyle='--',color="khaki")
curve = plots009('saved_models/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Anti-Curriculum',linestyle='--',color="lightsteelblue")
curve = plots009('saved_models/MIXTURE_BETA_8_VANILLA.h5','MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Vanilla',linestyle='--',color="mediumseagreen")
'''
curve = plots009('saved_models/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Curriculum',linestyle='--',color="mediumseagreen")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('s009_CURR_NON_LOCAL.png', dpi=150)
plt.clf()


curve = plots010('saved_models/MULTIMODAL_OLD_BETA_8_CURR.h5', 'MULTIMODAL_OLD')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Multimodal Old',linestyle='--',color="salmon")

curve = plots010('saved_models/NON_LOCAL_MIXTURE_BETA_8_CURR.h5', 'NON_LOCAL_MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Non Local Mixture',linestyle='--',color="lightsteelblue")
'''
curve = plots010('saved_models/MIXTURE_BETA_8_ONLY_LOS.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Only LOS',linestyle='--',color="darkgray")
curve = plots010('saved_models/MIXTURE_BETA_8_ONLY_NLOS.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Only NLOS',linestyle='--',color="khaki")
curve = plots010('saved_models/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Anti-Curriculum',linestyle='--',color="lightsteelblue")
curve = plots010('saved_models/MIXTURE_BETA_8_VANILLA.h5','MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Vanilla',linestyle='--',color="mediumseagreen")
'''
curve = plots010('saved_models/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label='Curriculum',linestyle='--',color="mediumseagreen")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s010')
plt.ylabel('top-K')
plt.grid()
plt.savefig('s010_CURR_NON_LOCAL.png', dpi=150)
