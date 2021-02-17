import numpy as np
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD, MIXTURE,NON_LOCAL_MIXTURE
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
def testModel(model,LIDAR_val,POS_val,Y_val):
    preds = model.predict([LIDAR_val,POS_val])    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve
def testModelNLOSLOS(model,LIDAR_val,POS_val,Y_val):
    preds = model.predict([LIDAR_val,POS_val])    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve
def testIPCModel(model,LIDAR_val,Y_val):
    preds = model.predict([LIDAR_val])    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve

def throughtputRatio(model,LIDAR_val,POS_val):
    _, _, Y_val, _ =load_dataset('./data/s009_unnormalized_labels.npz',True,False)
    preds = model.predict([LIDAR_val,POS_val])    # Get predictionspredictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros((len(preds),256))
    max_gain=np.zeros(len(preds))
    for i in range(0,len(preds)):
        max_gain[i]=Y_val[i,true[i]]
        curve[i,0]=Y_val[i,preds[i,0]]
        for j in range(1,256):
            curve[i,j]=np.max([curve[i,j-1],Y_val[i,preds[i,j]]])
    curve=np.sum(np.log2(1+curve),axis=0)/np.sum(np.log2(max_gain+1))
    return curve



def throughtputRatioIPC(model,LIDAR_val):
    _, _, Y_val, _ =load_dataset('./data/s009_unnormalized_labels.npz',True,False)
    preds = model.predict(LIDAR_val)    # Get predictionspredictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros((len(preds),256))
    max_gain=np.zeros(len(preds))
    for i in range(0,len(preds)):
        max_gain[i]=Y_val[i,true[i]]
        curve[i,0]=Y_val[i,preds[i,0]]
        for j in range(1,256):
            curve[i,j]=np.max([curve[i,j-1],Y_val[i,preds[i,j]]])
    curve=np.sum(np.log2(1+curve),axis=0)/np.sum(np.log2(max_gain+1))
    return curve

'''Training Parameters'''
PATH='Final'
MC_REPS=10
BETA=[0.8]    #Beta loss values to test
TEST_S010=False
VAL_S009=True
NET_TYPE = 'MIXTURE'    #Type of network
FLATTENED=True      #If True Lidar is 2D
SUM=False     #If True uses the method lidar_to_2d_summing() instead of lidar_to_2d() in dataLoader.py to process the LIDAR
SHUFFLE=False
LIDAR_TYPE='ABSOLUTE'
TRAIN_TYPES=['CURR','ANTI','VANILLA','ONLY_LOS','ONLY_NLOS']
TRAIN_TYPE='CURR'
if TRAIN_TYPE not in TRAIN_TYPES:
    print('Vanilla training over the entire dataset')
    TRAIN_TYPE=''
batch_size = 32
num_epochs = 45
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
POS_tr=POS_tr[:,0:2]
POS_val=POS_val[:,0:2]
POS_te=POS_te[:,0:2]
if(SHUFFLE):
    ind=np.random.shuffle(np.arange(Y_tr.shape[0])-1)
    POS_tr=POS_tr[ind,:][0]
    LIDAR_tr=LIDAR_tr[ind,:,:,:][0]
    Y_tr=Y_tr[ind,:][0]
    NLOS_tr=NLOS_tr[ind][0]
if TRAIN_TYPE in TRAIN_TYPES:
    stumps=5
    Perc=np.linspace(0,1,stumps)
    NLOSind = np.where(NLOS_tr == 0)[0]
    LOSind = np.where(NLOS_tr == 1)[0]
if(NET_TYPE=='MULTIMODAL' or NET_TYPE=='MULTIMODAL_OLD' or NET_TYPE=='MIXTURE' or NET_TYPE == "NON_LOCAL_MIXTURE"or NET_TYPE == "IPC"):
    print('Lidar scaled in 0-3 interval')
    LIDAR_tr = LIDAR_tr * 3 - 2
    LIDAR_val = LIDAR_val * 3 - 2
    LIDAR_te = LIDAR_te * 3 - 2

NLOSind_val=np.where(NLOS_val == 0)[0]
LOSind_val=np.where(NLOS_val == 1)[0]

for beta in BETA:
    seed=1
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if VAL_S009:
        checkpoint = ModelCheckpoint('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'.h5', monitor='val_top_10_accuracy', verbose=1,  save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)
    #Training Phase
    curves10=[]
    curves=[]
    curves_NLOS=[]
    curves_LOS=[]
    throughtput=[]
    if(NET_TYPE=='MULTIMODAL' or NET_TYPE=='MULTIMODAL_OLD' or NET_TYPE=='MIXTURE' or NET_TYPE == "NON_LOCAL_MIXTURE"):
        for rep in range(0,MC_REPS):
            optim = Adam(lr=1e-3, epsilon=1e-8)
            scheduler = lambda epoch, lr: lr
            callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            if(NET_TYPE=='MULTIMODAL'):
                model= MULTIMODAL(FLATTENED,LIDAR_TYPE)
            if(NET_TYPE=='MULTIMODAL_OLD'):
                model= MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE)
            elif (NET_TYPE == "NON_LOCAL_MIXTURE"):
                model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE)
            elif (NET_TYPE == 'MIXTURE'):
                model = MIXTURE(FLATTENED, LIDAR_TYPE)
                '''Mixture seems to work well on unnormalized'''
            model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
            print('MONTE-CARLO REP '+str(rep)+' OUT OF '+str(MC_REPS))
            if(rep==0):
                model.summary()
            if TRAIN_TYPE in TRAIN_TYPES:
                for ep in range(0,stumps):
                    if(TRAIN_TYPE=='CURR'):
                        ind=np.concatenate((np.random.choice(NLOSind, int((Perc[ep])*NLOSind.shape[0]),replace=False),np.random.choice(LOSind, LOSind.shape[0],replace=False)),axis=None)
                    elif(TRAIN_TYPE=='ANTI'):
                        ind=np.concatenate((np.random.choice(NLOSind,NLOSind.shape[0],replace=False),np.random.choice(LOSind,int(Perc[ep]*LOSind.shape[0]),replace=False)),axis=None)
                    elif(TRAIN_TYPE=='VANILLA'):
                        samples=LOSind.shape[0]+int(Perc[ep]*NLOSind.shape[0])
                        ind=np.random.choice(np.arange(0,LIDAR_tr.shape[0]),samples,replace=False)
                        #ind=np.concatenate((np.random.choice(NLOSind,int(0.5*samples)),np.random.choice(LOSind,int(0.5*samples))))
                    elif(TRAIN_TYPE=='ONLY_NLOS'):
                        ind=NLOSind
                    elif(TRAIN_TYPE=='ONLY_LOS'):
                        ind=LOSind
                    np.random.shuffle(ind)
                    if VAL_S009:
                        hist = model.fit([LIDAR_tr[ind,:,:,:],POS_tr[ind,:]], Y_tr[ind,:],validation_data=([LIDAR_val, POS_val], Y_val), epochs=int(num_epochs/stumps),batch_size=batch_size, callbacks=[checkpoint, callback])
                    else:
                        hist = model.fit([LIDAR_tr[ind,:,:,:],POS_tr[ind,:]], Y_tr[ind,:], epochs=int(num_epochs/stumps),batch_size=batch_size, callbacks=[callback])
                    if ep==0 and rep==0:
                        total_hist=hist
                    else:
                        for key in hist.history.keys():
                            total_hist.history[key].extend(hist.history[key])
                model.save_weights('./' + PATH + '/' + NET_TYPE + '_BETA_' + str(int(beta * 10)) + '_' + str(rep) + '_' + TRAIN_TYPE + '.h5')
                c = testModel(model, LIDAR_val, POS_val, Y_val)
                c_NLOS = testModel(model, LIDAR_val[NLOSind_val, :, :, :], POS_val[NLOSind_val, :], Y_val[NLOSind_val])
                curves_NLOS.append(c_NLOS)
                c_LOS = testModel(model, LIDAR_val[LOSind_val, :, :, :], POS_val[LOSind_val, :], Y_val[LOSind_val])
                curves_LOS.append(c_LOS)
                th = throughtputRatio(model, LIDAR_val, POS_val)
                throughtput.append(th)
                print('Throughput s009:')
                print('Top 1:'+str(th[0]))
                print('Top 5:'+str(th[4]))
                print('Top 10:'+str(th[9]))
                print('FINAL s009:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves.append(c)
                c=testModel(model,LIDAR_te,POS_te,Y_te)
                print('FINAL s010:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves10.append(c)
            else:
                if VAL_S009:
                    hist = model.fit([LIDAR_tr, POS_tr], Y_tr, validation_data=([LIDAR_val, POS_val], Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
                else:
                    hist = model.fit([LIDAR_tr, POS_tr], Y_tr, epochs=num_epochs, batch_size=batch_size,callbacks=[callback])
                if rep==0:
                    total_hist=hist
                else:
                    for key in hist.history.keys():
                        total_hist.history[key].extend(hist.history[key])
                model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+str(rep)+'_'+TRAIN_TYPE+'.h5')
                c=testModel(model,LIDAR_val,POS_val,Y_val)
                c_NLOS=testModel(model,LIDAR_val[NLOSind_val,:,:,:],POS_val[NLOSind_val,:],Y_val[NLOSind_val])
                curves_NLOS.append(c_NLOS)
                c_LOS=testModel(model,LIDAR_val[LOSind_val,:,:,:],POS_val[LOSind_val,:],Y_val[LOSind_val])
                curves_LOS.append(c_LOS)
                th=throughtputRatio(model,LIDAR_val,POS_val)
                throughtput.append(th)
                print('Throughput s009:')
                print('Top 1:'+str(th[0]))
                print('Top 5:'+str(th[4]))
                print('Top 10:'+str(th[9]))
                print('FINAL s009:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves.append(c)
                c=testModel(model,LIDAR_te,POS_te,Y_te)
                print('FINAL s010:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves10.append(c)
        np.save('./'+PATH+'/Curves'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'010', curves10)
        np.save('./' + PATH + '/Curves' + NET_TYPE + '_BETA_' + str(int(beta * 10)) + '_' + TRAIN_TYPE,curves)
        np.save('./'+PATH+'/CurvesLOS'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE, curves_NLOS)
        np.save('./' + PATH + '/CurvesNLOS' + NET_TYPE + '_BETA_' + str(int(beta * 10)) + '_' + TRAIN_TYPE, curves_LOS)
        np.save('./'+PATH+'/CurvesTH'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE, throughtput)
        model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+str(rep)+'_'+TRAIN_TYPE+'.h5')
        print(NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'     CURVE  SAVED!')
    elif(NET_TYPE=='IPC'):
        for rep in range(0,MC_REPS):
            optim = Adam(lr=1e-3, epsilon=1e-8)
            scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10.
            callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            model= LIDAR(FLATTENED,LIDAR_TYPE)
            model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
            if(rep==0):
                model.summary()
            if TRAIN_TYPE in TRAIN_TYPES:
                for ep in range(0, stumps):
                    if(TRAIN_TYPE=='CURR'):
                        ind=np.concatenate((np.random.choice(NLOSind, int((Perc[ep])*NLOSind.shape[0])),np.random.choice(LOSind, LOSind.shape[0])),axis=None)
                    elif(TRAIN_TYPE=='ANTI'):
                        ind=np.concatenate((np.random.choice(NLOSind,NLOSind.shape[0]),np.random.choice(LOSind,int(Perc[ep]*LOSind.shape[0]))),axis=None)
                    elif(TRAIN_TYPE=='VANILLA'):
                        samples=LOSind.shape[0]+int(Perc[ep]*NLOSind.shape[0])
                        ind=np.random.choice(np.arange(0,LIDAR_tr.shape[0]),samples)
                    np.random.shuffle(ind)
                    if VAL_S009:
                        hist = model.fit(LIDAR_tr[ind, :, :, :], Y_tr[ind, :],validation_data=(LIDAR_val, Y_val), epochs=int(num_epochs/stumps), batch_size=batch_size,callbacks=[checkpoint, callback])
                    else:
                        hist = model.fit(LIDAR_tr[ind, :, :, :], Y_tr[ind, :], epochs=int(num_epochs/stumps), batch_size=batch_size,callbacks=[ callback])
                    if ep==0 and rep==0:
                        total_hist=hist
                    else:
                        for key in hist.history.keys():
                            total_hist.history[key].extend(hist.history[key])
                model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+str(rep)+'_'+TRAIN_TYPE+'.h5')
                c=testIPCModel(model,LIDAR_val,Y_val)
                print('FINAL s009:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves.append(c)
                c=testIPCModel(model,LIDAR_te,Y_te)
                print('FINAL s010:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
            else:
                hist = model.fit(LIDAR_tr, Y_tr, validation_data=(LIDAR_val, Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[callback])
                if ep==0:
                    total_hist=hist
                else:
                    for key in hist.history.keys():
                        total_hist.history[key].extend(hist.history[key])
                model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+str(rep)+'_'+TRAIN_TYPE+'.h5')
                c=testIPCModel(model,LIDAR_val,Y_val)
                print('FINAL s009:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
                curves.append(c)
                c=testIPCModel(model,LIDAR_te,Y_te)
                print('FINAL s010:')
                print('Top 1:'+str(c[0]/c[len(c)-1]))
                print('Top 5:'+str(c[4]/c[len(c)-1]))
                print('Top 10:'+str(c[9]/c[len(c)-1]))
        np.save('./'+PATH+'/Curves'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'010', curves10)
        np.save('./'+PATH+'/Curves'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE, curves)
        model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+rep+'_'+TRAIN_TYPE+'.h5')
        print(NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'     CURVE  SAVED!')
    elif (NET_TYPE == 'GPS'):
        for rep in range(0,MC_REPS):
            model= GPS()
            model.compile(loss=KDLoss(beta),optimizer=optim,metrics=[metrics.categorical_accuracy,top_5_accuracy,top_10_accuracy,top_50_accuracy])
            if(rep==0):
                model.summary()
            if TRAIN_TYPE in TRAIN_TYPES:
                for ep in range(0, stumps):
                    if(TRAIN_TYPE=='CURR'):
                        ind=np.concatenate((np.random.choice(NLOSind, int((Perc[ep])*NLOSind.shape[0])),np.random.choice(LOSind, LOSind.shape[0])),axis=None)
                    elif(TRAIN_TYPE=='ANTI'):
                        ind=np.concatenate((np.random.choice(NLOSind,NLOSind.shape[0]),np.random.choice(LOSind,int(Perc[ep]*LOSind.shape[0]))),axis=None)
                    elif(TRAIN_TYPE=='VANILLA'):
                        samples=LOSind.shape[0]+int(Perc[ep]*NLOSind.shape[0])
                        ind=np.concatenate((np.random.choice(NLOSind,int(0.5*samples)),np.random.choice(LOSind,int(0.5*samples))))
                    np.random.shuffle(ind)
                    hist = model.fit(POS_tr[ind,:], Y_tr[ind, :], validation_data=(POS_val, Y_val), epochs=int(num_epochs/stumps), batch_size=batch_size,callbacks=[checkpoint, callback])
                    if ep==0 and rep==0:
                        total_hist=hist
                    else:
                        for key in hist.history.keys():
                            total_hist.history[key].extend(hist.history[key])
            else:
                hist = model.fit(POS_tr, Y_tr, validation_data=(POS_val, Y_val), epochs=num_epochs, batch_size=batch_size,callbacks=[checkpoint, callback])
                if rep==0:
                    total_hist=hist
                else:
                    for key in hist.history.keys():
                        total_hist.history[key].extend(hist.history[key])
    '''Saving weights and history for later'''
    if(TEST_S010):
        model.save_weights('./'+PATH+'/'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_FINAL_'+TRAIN_TYPE+'.h5')
        with open('./'+PATH+'/History'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE, 'wb') as file_pi:
            pickle.dump(total_hist.history, file_pi)
        model.load_weights('./'+PATH+'/' + NET_TYPE + '_BETA_' + str(int(beta * 10))+'_'+TRAIN_TYPE+'.h5')
        '''Testing on sS010'''

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
        np.savetxt('./'+PATH+'/PREDS_'+NET_TYPE+'_BETA_'+str(int(beta*10))+'_'+TRAIN_TYPE+'.csv', pred, delimiter=',')
