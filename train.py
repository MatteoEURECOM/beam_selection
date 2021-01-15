import numpy as np
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD, MIXTURE
from dataLoader import load_dataset
import pickle
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam
import matplotlib.pyplot  as plt

### Metrics ###
def top_5_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)
def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)
def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

### CUSTOM LOSS;  mixing cross-entropy with squashed and soft probabilities ###
def KDLoss(beta):
    def loss(y_true,y_pred):
        y_true_hard = tf.one_hot(tf.argmax(y_true, axis = 1), depth = 256)
        kl = tf.keras.losses.KLDivergence()
        return beta*kl(y_true,y_pred)+(1-beta)*kl(y_true_hard,y_pred)
    return loss

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


'''Training Parameters'''
BETA=[0.2,0.4,0.6,0.8,1]    #Beta loss values to test
CURRICULUM=True    #If True starts increases the NLOS samples percentage in the epoch accoring to the Perc array
SAVE_INIT=True      #Use the same weights initialization each time beta is updated
NET_TYPE = 'MIXTURE'    #Type of network
FLATTENED=True      #If True Lidar is 2D
SUM=False       #If True uses the method lidar_to_2d_summing() instead of lidar_to_2d() in dataLoader.py to process the LIDAR
SHUFFLE=False
LIDAR_TYPE='ABSOLUTE'   #Type of lidar images CENTERED: lidar centered at Rx, ABSOLUTE: lidar images as provided  and ABSOLUTE_LARGE: lidar images of larger size
seed=1
np.random.seed(seed)
tf.random.set_seed(seed)
batch_size = 32
num_epochs = 30

'''Loading Data'''
if LIDAR_TYPE=='CENTERED':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008_centered.npz',FLATTENED,SUM)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED,SUM)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010_centered.npz',FLATTENED,SUM)
elif LIDAR_TYPE=='ABSOLUTE':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008_original_labels.npz',FLATTENED,SUM)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_original_labels.npz',FLATTENED,SUM)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010_original_labels.npz',FLATTENED,SUM)
elif LIDAR_TYPE=='ABSOLUTE_LARGE':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008_large.npz',FLATTENED,SUM)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_large.npz',FLATTENED,SUM)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010_large.npz',FLATTENED,SUM)
if(SHUFFLE):
    ind=np.random.shuffle(np.arange(Y_tr.shape[0])-1)
    POS_tr=POS_tr[ind,:][0]
    LIDAR_tr=LIDAR_tr[ind,:,:,:][0]
    Y_tr=Y_tr[ind,:][0]
    NLOS_tr=NLOS_tr[ind][0]
print(np.sum(NLOS_val))
print(np.sum(NLOS_tr))
if(False):
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(np.squeeze(np.mean(LIDAR_tr, axis=0)))
    axarr[1].imshow(np.squeeze(np.mean(LIDAR_val, axis=0)))
    axarr[2].imshow(np.squeeze(np.mean(LIDAR_te, axis=0)))
    plt.show()
    LIDAR_tr[LIDAR_tr < 0.6] = 0
    LIDAR_tr[LIDAR_tr > 0.8] = 0
    LIDAR_val[LIDAR_val < 0.6] = 0
    LIDAR_val[LIDAR_val > 0.8] = 0
    LIDAR_te[LIDAR_te < 0.6] = 0
    LIDAR_te[LIDAR_te > 0.8] = 0
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(np.squeeze(np.mean(LIDAR_tr, axis=0)))
    axarr[1].imshow(np.squeeze(np.mean(LIDAR_val, axis=0)))
    axarr[2].imshow(np.squeeze(np.mean(LIDAR_te, axis=0)))
    plt.show()

if CURRICULUM :
    num_epochs=int(num_epochs*1.25)
    Perc=np.concatenate([np.linspace(0,1,int(num_epochs/2)),np.ones(num_epochs-int(num_epochs/2))])
    NLOSind = np.where(NLOS_tr == 0)[0]
    LOSind = np.where(NLOS_tr == 1)[0]

#Initializing the model
if(NET_TYPE=='MULTIMODAL'):
    model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
if(NET_TYPE=='MULTIMODAL_OLD'):
    model= MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE)
elif(NET_TYPE=='IPC'):
    model= LIDAR(FLATTENED,LIDAR_TYPE)
elif (NET_TYPE == 'GPS'):
    model = GPS()
elif (NET_TYPE == 'MIXTURE'):
    model = MIXTURE(FLATTENED, LIDAR_TYPE)
    LIDAR_tr = LIDAR_tr * 3 - 2
    LIDAR_val = LIDAR_val * 3 - 2
    LIDAR_te = LIDAR_te * 3 - 2

for beta in BETA:
    optim = Adam(lr=1e-3, epsilon=1e-8)
    scheduler = lambda epoch, lr: lr if NET_TYPE == 'MIXTURE' else lambda epoch, lr: lr if epoch < 10 else lr*tf.math.exp(-0.1)
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
    model.summary()

    if(SAVE_INIT):
        INIT_WEIGHTS=model.get_weights()
        SAVE_INIT=False
    else:
        model.set_weights(INIT_WEIGHTS)
    if(CURRICULUM):
        checkpoint = ModelCheckpoint('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'CURRICULUM.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)
    else:
        checkpoint = ModelCheckpoint('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)
    #Training Phase
    if(NET_TYPE=='MULTIMODAL' or NET_TYPE=='MIXTURE'):
        if(CURRICULUM):
            for ep in range(0,num_epochs):
                ind=np.concatenate((np.random.choice(NLOSind, int((Perc[ep])*NLOSind.shape[0])),np.random.choice(LOSind, LOSind.shape[0])),axis=None)
                np.random.shuffle(ind)
                hist = model.fit([LIDAR_tr[ind,:,:,:],POS_tr[ind,:]], Y_tr[ind,:],validation_data=([LIDAR_val, POS_val], Y_val), epochs=1,batch_size=batch_size, callbacks=[checkpoint, callback])

        else:
            hist = model.fit([LIDAR_tr,POS_tr], Y_tr, validation_data=([LIDAR_val,POS_val], Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif(NET_TYPE=='IPC'):
        if (CURRICULUM):
            for ep in range(0, num_epochs):
                #ind = np.concatenate((np.random.choice(NLOSind, int(Perc[ep] * data_size_curr)),np.random.choice(LOSind, int((1 - Perc[ep]) * data_size_curr))), axis=None)
                ind = np.concatenate((np.random.choice(NLOSind, int((Perc[ep]) * NLOSind.shape[0])),np.random.choice(LOSind, LOSind.shape[0])), axis=None)
                hist = model.fit(LIDAR_tr[ind, :, :, :], Y_tr[ind, :],validation_data=(LIDAR_val, Y_val), epochs=1, batch_size=batch_size,callbacks=[checkpoint, callback])
        else:
            hist = model.fit(LIDAR_tr, Y_tr, validation_data=(LIDAR_val, Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif (NET_TYPE == 'GPS'):
        if (CURRICULUM):
            for ep in range(0, num_epochs):
                #ind = np.concatenate((np.random.choice(NLOSind, int(Perc[ep] * data_size_curr)),np.random.choice(LOSind, int((1 - Perc[ep]) * data_size_curr))), axis=None)
                ind = np.concatenate((np.random.choice(NLOSind, int((Perc[ep]) * NLOSind.shape[0])),np.random.choice(LOSind, LOSind.shape[0])), axis=None)
                hist = model.fit(POS_tr[ind,:], Y_tr[ind, :], validation_data=(POS_val, Y_val), epochs=1, batch_size=batch_size,callbacks=[checkpoint, callback])
        else:
            hist = model.fit(POS_tr, Y_tr, validation_data=(POS_val, Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])

    #Saving weights and history for later
    if(CURRICULUM):
        model.save_weights('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'FINAL_CURRICULUM.h5')
        with open('./saved_models/History'+NET_TYPE+'_BETA_'+str(int(beta*10))+'CURRICULUM', 'wb') as file_pi:
         pickle.dump(hist.history, file_pi)
        model.load_weights('./saved_models/' + NET_TYPE + '_BETA_' + str(int(beta * 10))+'CURRICULUM.h5')
    else:
        model.save_weights('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'FINAL.h5')
        with open('./saved_models/History'+NET_TYPE+'_BETA_'+str(int(beta*10)), 'wb') as file_pi:
           pickle.dump(hist.history, file_pi)
        model.load_weights('./saved_models/' + NET_TYPE + '_BETA_' + str(int(beta * 10))+'.h5')

    #Testing phase on s010
    if(NET_TYPE=='MULTIMODAL' or NET_TYPE=='MIXTURE'):
        preds = model.predict([LIDAR_te,POS_te])
    elif(NET_TYPE=='IPC'):
        preds = model.predict([LIDAR_te])
    elif (NET_TYPE == 'GPS'):
        preds = model.predict([POS_te])
    num_rows = 8
    num_columns = 32
    to_save=[]
    for data in preds:
        if len(data) != (num_rows * num_columns):
            raise Exception('Number of elements in this row is not the product num_rows * num_columns')
        new_vector = reorder(data, num_rows, num_columns)
        to_save.append(new_vector)
    to_save=np.asarray(to_save)
    pred= np.argsort(-to_save, axis=1) #Descending order
    if(CURRICULUM):
        np.savetxt('./saved_models/PREDS_'+NET_TYPE+'_BETA_'+str(int(beta*10))+'CURRICULUM.csv', pred, delimiter=',')
    else:
        np.savetxt('./saved_models/PREDS_'+NET_TYPE+'_BETA_'+str(int(beta*10))+'.csv', pred, delimiter=',')
