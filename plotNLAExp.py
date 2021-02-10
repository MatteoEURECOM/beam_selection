import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD,MIXTURE,NON_LOCAL_MIXTURE
import tensorflow as tf
import keras
from utils import plots009,plots010,finalMCAvg



#METRICS Plot


val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('NLAExp/HistoryNON_LOCAL_MIXTURE_BETA_8_CURREMBEEDED',5)
plt.errorbar([1.1,2.1,3.1], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'embedded',linestyle='None', marker='.',capsize=2,color="orangered")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('NLAExp/HistoryNON_LOCAL_MIXTURE_BETA_8_CURRGAUSS',5)
plt.errorbar([1.3,2.3,3.3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'gaussian',linestyle='None', marker='.',capsize=2,color="khaki")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('NLAExp/HistoryNON_LOCAL_MIXTURE_BETA_8_CURR',5)
plt.errorbar([1.4,2.4,3.4], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'dot',linestyle='None', marker='.',capsize=2,color="yellowgreen")

axes = plt.gca()
axes.set_yticks(np.arange(0.3, 1, 0.05))
axes.set_yticks(np.arange(0.3, 1, 0.01), minor=True)
axes.set_xticks([1.2,2.2,3.2])
axes.set_xticklabels(['Top-1','Top-5','Top-10'])
axes.grid(which='minor', alpha=0.2)
axes.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig('NLAExp/Metrics.pdf')
plt.show()
plt.clf()

#Top-K s009 Plot

curve = plots009('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURREMBEEDED.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','embedded',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'embedded',linestyle='--',color="red")
curve = plots009('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURRGAUSS.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','gaussian',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'gaussian',linestyle='--',color="orangered")
curve=plots009('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR_DOT.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot',linestyle='--',color="orange")
curve = plots009('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR_DOT_2.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot 2',linestyle='--',color="khaki")
curve = plots009('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot 4',linestyle='--',color="yellowgreen")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('NLAExp/s009.pdf')
plt.clf()

#Top-K s010 Plot

curve = plots010('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURREMBEEDED.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','embedded',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'embedded',linestyle='--',color="red")
curve = plots010('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURRGAUSS.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','gaussian',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'gaussian',linestyle='--',color="orangered")
curve = plots010('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR_DOT.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot',linestyle='--',color="orange")
curve = plots010('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR_DOT_2.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot 2',linestyle='--',color="khaki")
curve = plots010('NLAExp/NON_LOCAL_MIXTURE_BETA_8_CURR.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'dot 4',linestyle='--',color="yellowgreen")

plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s010')
plt.ylabel('top-K')
plt.grid()
plt.savefig('NLAExp/s010.pdf')
