import numpy as np
from models import MULTIMODAL,LIDAR2D,GPS,MULTIMODAL_OLD
from dataLoader import load_dataset
import pickle
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam

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
        return beta*categorical_crossentropy(y_true,y_pred)+(1-beta)*categorical_crossentropy(y_true_hard,y_pred)
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


#Training Parameters
CURRICULUM= False
SAVE_INIT=True
NET_TYPE = 'MULTIMODAL'
FLATTENED=True
LIDAR_TYPE='ABSOLUTE'
np.random.seed(2021)
batch_size = 32
num_epochs = 15
learning_rate=0.0001
#Loading Data
if LIDAR_TYPE=='CENTERED':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008_centered.npz',FLATTENED)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_centered.npz',FLATTENED)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010_centered.npz',FLATTENED)
elif LIDAR_TYPE=='ABSOLUTE':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008.npz',FLATTENED)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009.npz',FLATTENED)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010.npz',FLATTENED)
elif LIDAR_TYPE=='ABSOLUTE_LARGE':
    POS_tr, LIDAR_tr, Y_tr, NLOS_tr = load_dataset('./data/s008_large.npz',FLATTENED)
    POS_val, LIDAR_val, Y_val, NLOS_val =load_dataset('./data/s009_large.npz',FLATTENED)
    POS_te, LIDAR_te, Y_te, _ =load_dataset('./data/s010_large.npz',FLATTENED)
if CURRICULUM :
    data_size_curr=10000
    Perc=np.linspace(0.1,0.9,num_epochs)
    NLOSind = np.where(NLOS_tr == 0)[0]
    LOSind = np.where(NLOS_tr == 1)[0]
#Initializing the model
if(NET_TYPE=='MULTIMODAL'):
    model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
if(NET_TYPE=='MULTIMODAL_OLD'):
    model= MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE)
elif(NET_TYPE=='IPC'):
    model= LIDAR2D(FLATTENED,LIDAR_TYPE)
elif (NET_TYPE == 'GPS'):
    model = GPS(FLATTENED,LIDAR_TYPE)
optim = Adam(lr=learning_rate)
scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10.
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

for beta in [0.8]:
    model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
    model.summary()
    if(SAVE_INIT):
        INIT_WEIGHTS=model.get_weights()
        SAVE_INIT=False
    else:
        model.set_weights(INIT_WEIGHTS)
    checkpoint = ModelCheckpoint('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)
    #Training Phase
    if(NET_TYPE=='MULTIMODAL'):
        if(CURRICULUM):
            for ep in range(0,num_epochs):
                ind=np.concatenate((np.random.choice(NLOSind, int(Perc[ep]*data_size_curr)),np.random.choice(LOSind, int((1-Perc[ep])*data_size_curr))),axis=None)
                model.fit([LIDAR_tr[ind,:,:,:],Y_tr[ind,:]], Y_tr[ind,:],validation_data=([LIDAR_te, Y_te], Y_te), epochs=1,batch_size=batch_size, callbacks=[checkpoint, callback])
        else:
            hist = model.fit([LIDAR_tr,POS_tr], Y_tr, validation_data=([LIDAR_te,POS_te], Y_te), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif(NET_TYPE=='IPC'):
        if (CURRICULUM):
            for ep in range(0, num_epochs):
                ind = np.concatenate((np.random.choice(NLOSind, int(Perc[ep] * data_size_curr)),np.random.choice(LOSind, int((1 - Perc[ep]) * data_size_curr))), axis=None)
                model.fit(LIDAR_tr[ind, :, :, :], Y_tr[ind, :],validation_data=(LIDAR_te, Y_te), epochs=1, batch_size=batch_size,callbacks=[checkpoint, callback])
        else:
            hist = model.fit(LIDAR_tr, Y_tr, validation_data=(LIDAR_te, Y_te), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif (NET_TYPE == 'GPS'):
        if (CURRICULUM):
            for ep in range(0, num_epochs):
                ind = np.concatenate((np.random.choice(NLOSind, int(Perc[ep] * data_size_curr)),np.random.choice(LOSind, int((1 - Perc[ep]) * data_size_curr))), axis=None)
                hist = model.fit(POS_tr[ind,:], Y_tr, validation_data=(POS_te[ind,:], Y_te), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
        else:
            hist = model.fit(POS_tr, Y_tr, validation_data=(POS_te, Y_te), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    #Saving weights and history for later
    model.save_weights('./saved_models/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'FINAL.h5')
    with open('./saved_models/History'+NET_TYPE+'_BETA_'+str(int(beta*10)), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    model.load_weights('./saved_models/' + NET_TYPE + '_BETA_' + str(int(beta * 10))+'.h5')
    #Testing phase on s010
    if(NET_TYPE=='MULTIMODAL'):
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
    np.savetxt('./saved_models/PREDS_'+NET_TYPE+'_BETA_'+str(int(beta*10))+'.csv', pred, delimiter=',')