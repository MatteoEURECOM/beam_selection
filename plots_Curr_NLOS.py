import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.io
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD, MIXTURE,NON_LOCAL_MIXTURE
import pandas
import pickle


with open('lasse_preds_009.pkl', 'rb') as f:
    results = pickle.load(f)
    preds_idx = results['preds_idx']
    raw_preds = results['raw_preds']

NLOS=True
Create=False
Plot=True
def throughtputRatio(preds,Y_val):
    test_preds_idx = np.argsort(preds, axis=1)
    throughput_ratio_at_k = np.zeros(100)
    for i in range(100):
        throughput_ratio_at_k[i] = np.sum(np.log2(np.max(np.take_along_axis(Y_val, test_preds_idx, axis=1)[:, -1 - i:], axis=1) + 1.0)) / np.sum(np.log2(np.max(Y_val, axis=1) + 1.0))
    return throughput_ratio_at_k


def throughtputRatioTest(preds,Y_val):   # Get predictionspredictions
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
def testModel(preds,Y_val):    # Get predictions
    preds= np.argsort(-preds, axis=1) #Descending order
    true=np.argmax(Y_val[:,:], axis=1) #Best channel
    curve=np.zeros(256)
    for i in range(0,len(preds)):
        curve[np.where(preds[i,:] == true[i])]=curve[np.where(preds[i,:] == true[i])]+1
    curve=np.cumsum(curve)
    return curve

if(Create):
    FLATTENED=True      #If True Lidar is 2D
    SUM=False     #If True uses the method lidar_to_2d_summing() instead of lidar_to_2d() in dataLoader.py to process the LIDAR
    LIDAR_TYPE='ABSOLUTE'
    POS_te, LIDAR_te, Y_te, NLOS_te =load_dataset('./data/s008_original_labels.npz',FLATTENED,SUM)
    LIDAR_te = LIDAR_te * 3 - 2
    #Y_te= np.abs(np.load('./data/beams_output_test.npz')['output_classification']).reshape(-1,256)

    for beta in [8]:
        if (NLOS):
            th_LOS = []
            acc_LOS = []
            th_NLOS = []
            acc_NLOS = []
            NLOSind_te = np.where(NLOS_te == 0)[0]
            LOSind_te = np.where(NLOS_te == 1)[0]
        else:
            th = []
            acc = []
        for k in range(0,10):
            model = NON_LOCAL_MIXTURE(FLATTENED, LIDAR_TYPE)
            model.load_weights('./Final/NON_LOCAL_MIXTURE_BETA_'+str(beta)+'_'+str(k)+'_CURR.h5')
            model.summary()
            if NLOS:
                LOS_preds=model.predict([LIDAR_te[LOSind_te, :, :, :], POS_te[LOSind_te, 0:2]])
                NLOS_preds=model.predict([LIDAR_te[NLOSind_te, :, :, :], POS_te[NLOSind_te, 0:2]])
                th_LOS.append(throughtputRatio(LOS_preds,Y_te[LOSind_te,:]))
                acc_LOS.append(testModel(LOS_preds, Y_te[LOSind_te,:]))
                th_NLOS.append(throughtputRatio(NLOS_preds, Y_te[NLOSind_te,:]))
                acc_NLOS.append(testModel(NLOS_preds, Y_te[NLOSind_te,:]))
            else:
                preds=model.predict([LIDAR_te, POS_te[:, 0:2]])
                th.append(throughtputRatio(preds,Y_te))
                #acc.append(testModel(preds,Y_te))
            print(k)

        if NLOS:
            np.save('./Final/Curves_NON_LOCAL_MIXTURE_BETA_'+str(beta)+'_CURR008_LOS', acc_LOS)
            np.save('./Final/Curves_NON_LOCAL_MIXTURE_BETA_'+str(beta)+'_CURR008_NLOS', acc_NLOS)
            np.save('./Final/CurvesTH_NON_LOCAL_MIXTURE_BETA_'+str(beta)+'_CURR008_LOS', th_LOS)
            np.save('./Final/CurvesTH_NON_LOCAL_MIXTURE_BETA_'+str(beta)+'_CURR008_NLOS', th_NLOS)
        else:
            #np.save('./FinalGauss/CurvesMIXTURE_BETA_8_010', acc)
            np.save('./Final/CurvesTHMIXTURE_BETA_'+str(beta)+'_VANILLA010', th)
if(Plot):


    curve = np.load('Final/Curves_MIXTURE_BETA_8_CURR_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/Curves_MIXTURE_BETA_8_CURR_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1_NLOS = np.mean(curve, axis=0)



    curve = np.load('Final/Curves_MIXTURE_BETA_8_VANILLA_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/Curves_MIXTURE_BETA_8_VANILLA_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2_NLOS = np.mean(curve, axis=0)

    curve = np.load('Final/Curves_NON_LOCAL_MIXTURE_BETA_8_CURR_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/Curves_NON_LOCAL_MIXTURE_BETA_8_CURR_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3_NLOS = np.mean(curve, axis=0)

    plt.rc('font', family='serif', serif='Computer Modern Roman', size=14)
    plt.rc('text', usetex=True)


    fig, axs = plt.subplots(2)
    fig.set_figheight(8)
    axs[0].plot(np.arange(1, 31), data_y_1_LOS[0:30], marker='^', linewidth=1, markevery=1, markersize=4)
    axs[0].plot(np.arange(1, 31), data_y_2_LOS[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
    axs[0].plot(np.arange(1, 31), data_y_3_LOS[0:30], marker='v', linewidth=1, markevery=1,markersize=4)
    axs[0].plot(np.arange(1, 31), data_y_1_NLOS[0:30], marker='^', linewidth=1,linestyle='--' ,markevery=1, markersize=4, color='tab:blue')
    axs[0].plot(np.arange(1, 31), data_y_2_NLOS[0:30], marker='o', linewidth=1,linestyle='--', markevery=1, markersize=4,color='tab:orange')
    axs[0].plot(np.arange(1, 31), data_y_3_NLOS[0:30], marker='v', linestyle='--', linewidth=1, markevery=1, markersize=4,color='tab:green')
    axs[0].grid()
    axs[0].legend(['Mixture Curr LOS','Mixture LOS', 'NLA Curr LOS', 'Mixture Curr NLOS', 'Mixture NLOS','NLA Curr NLOS'], fontsize=14, loc='lower right')
    axs[0].set_xlabel("$k$", fontsize=14)
    axs[0].set_ylabel("Top-$k$ Accuracy", fontsize=14)
    axs[0].set_xlim([0, 30])

    curve = np.load('Final/CurvesTH_MIXTURE_BETA_8_CURR_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/CurvesTH_MIXTURE_BETA_8_CURR_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1_NLOS = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTH_MIXTURE_BETA_8_VANILLA_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/CurvesTH_MIXTURE_BETA_8_VANILLA_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2_NLOS = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTH_NON_LOCAL_MIXTURE_BETA_8_CURR_LOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3_LOS = np.mean(curve, axis=0)
    curve = np.load('Final/CurvesTH_NON_LOCAL_MIXTURE_BETA_8_CURR_NLOS.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3_NLOS = np.mean(curve, axis=0)

    axs[1].plot(np.arange(1, 31), data_y_1_LOS[0:30], marker='^', linewidth=1, markevery=1, markersize=4)
    axs[1].plot(np.arange(1, 31), data_y_2_LOS[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
    axs[1].plot(np.arange(1, 31), data_y_3_LOS[0:30], marker='v', linewidth=1, markevery=1,markersize=4)
    axs[1].plot(np.arange(1, 31), data_y_1_NLOS[0:30], marker='^', linewidth=1,linestyle='--' ,markevery=1, markersize=4, color='tab:blue')
    axs[1].plot(np.arange(1, 31), data_y_2_NLOS[0:30], marker='o', linewidth=1,linestyle='--', markevery=1, markersize=4,color='tab:orange')
    axs[1].plot(np.arange(1, 31), data_y_3_NLOS[0:30], marker='v', linestyle='--', linewidth=1, markevery=1, markersize=4,color='tab:green')
    axs[1].grid()
    axs[1].legend(['Mixture Curr LOS','Mixture LOS', 'NLA Curr LOS', 'Mixture Curr NLOS', 'Mixture NLOS','NLA Curr NLOS'], fontsize=14, loc='lower right')
    axs[1].set_xlabel("$k$", fontsize=14)
    axs[1].set_ylabel("Top-$k$ Throughput Ratio", fontsize=14)
    axs[1].set_xlim([0, 30])
    plt.savefig("CurrNLOSLOS.pdf")
