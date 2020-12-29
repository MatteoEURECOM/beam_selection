import numpy as np
from models import multiModalNet,Lidar2D,GPS
from dataLoader import loadDataTraining,loadDataValidation,loadDataTesting
from utils import reorder
import matplotlib.pyplot as plt
import csv,pickle,h5py, sys, os, shutil
import tensorflow as tf
import keras as K
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
SAVE_INIT=True
np.random.seed(2021)
Net = 'IPC'
num_epochs = 10
batch_size = 16
for beta in [0.2,0.4,0.6,0.8,1]:
    X_p_val,X_l_val,Y_val=loadDataValidation()
    X_p_train,X_l_train,Y_train=loadDataTraining()
    X_p_test,X_l_test,Y_test=loadDataTesting()
    opt = Adam()
    optim = Adam(lr=1e-3, epsilon=1e-8)
    scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10.
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    if(Net=='MULTIMODAL'):
        model= multiModalNet()
    elif(Net=='IPC'):
        model= Lidar2D
    elif (Net == 'GPS'):
        model = GPS
    model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
    model.summary()
    if(SAVE_INIT):
        INIT_WEIGHTS=model.get_weights()
        SAVE_INIT=False
    else:
        model.set_weights(INIT_WEIGHTS)
    checkpoint = ModelCheckpoint('./saved_models/'+Net+'_BETA_'+str(int(beta*10))+'.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)

    '''Training phase on s008 with validation on s009'''
    if(Net=='MULTIMODAL'):
        hist = model.fit([X_l_train,X_p_train[:,0:3]], Y_train, validation_data=([X_l_test,X_p_test[:,0:3]], Y_test), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif(Net=='IPC'):
        hist = model.fit(X_l_train, Y_train, validation_data=(X_l_test, Y_test), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    elif (Net == 'GPS'):
        hist = model.fit(X_p_train[:,0:3], Y_train, validation_data=(X_p_test[:,0:3], Y_test), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
    model.save_weights('./saved_models/'+Net+'_BETA_'+str(int(beta*10))+'FINAL.h5')
    with open('./saved_models/History'+Net+'_BETA_'+str(int(beta*10)), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    model.load_weights('./saved_models/' + Net + '_BETA_' + str(int(beta * 10))+'.h5')
    '''Testing phase on s010'''
    if(Net=='MULTIMODAL'):
        preds = model.predict([X_l_test,X_p_test[:,0:3]])
    elif(Net=='IPC'):
        preds = model.predict([X_l_test])
    elif (Net == 'GPS'):
        preds = model.predict([X_p_test[:,0:3]])
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
    np.savetxt('./saved_models/PREDS_'+Net+'_BETA_'+str(int(beta*10))+'.csv', pred, delimiter=',', fmt='%s')
