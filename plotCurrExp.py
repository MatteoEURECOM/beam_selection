import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD,MIXTURE,NON_LOCAL_MIXTURE
import tensorflow as tf
import keras
from utils import plots009,plots010,finalMCAvg,plotNLOSvsLOS


#Top-K s009 Plot NLOS vs LOS


curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curveLOS=curveLOS/ curveLOS[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curveNLOS[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Anti-Curriculum LOS',linestyle='--',color="orangered")
plt.plot(range(0, 256), curveNLOS, label=r'Anti-Curriculum NLOS',linestyle='-.',color="orangered")
curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_VANILLA.h5', 'MIXTURE')
curveLOS=curveLOS/ curveLOS[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curveNLOS[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Standard LOS',linestyle='--',color="khaki")
plt.plot(range(0, 256), curveNLOS, label=r'Standard NLOS',linestyle='-.',color="khaki")
curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curveLOS=curveLOS/ curveLOS[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curveNLOS[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Curriculum LOS',linestyle='--',color="yellowgreen")
plt.plot(range(0, 256), curveNLOS, label=r'Curriculum NLOS',linestyle='-.',color="yellowgreen")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('CurrExp/s009NLOSLOS.pdf')
plt.clf()


#METRICS Plot


val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('CurrExp/HistoryMIXTURE_BETA_8_ANTI',5)
plt.errorbar([1.1,2.1,3.1], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'Anti-Curriculum',linestyle='None', marker='.',capsize=2,color="orangered")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('CurrExp/HistoryMIXTURE_BETA_8_VANILLA',5)
plt.errorbar([1.3,2.3,3.3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'Standard',linestyle='None', marker='.',capsize=2,color="khaki")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('CurrExp/HistoryMIXTURE_BETA_8_CURR',5)
plt.errorbar([1.4,2.4,3.4], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'Curriculum',linestyle='None', marker='.',capsize=2,color="yellowgreen")

axes = plt.gca()
axes.set_yticks(np.arange(0.3, 1, 0.05))
axes.set_yticks(np.arange(0.3, 1, 0.01), minor=True)
axes.set_xticks([1.2,2.2,3.2])
axes.set_xticklabels(['Top-1','Top-5','Top-10'])
axes.grid(which='minor', alpha=0.2)
axes.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig('CurrExp/Metrics.pdf')
plt.clf()

#Top-K s009 Plot


curve = plots009('CurrExp/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Anti-Curriculum',linestyle='--',color="orangered")
curve = plots009('CurrExp/MIXTURE_BETA_8_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Standard',linestyle='--',color="khaki")
curve = plots009('CurrExp/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Curriculum',linestyle='--',color="yellowgreen")

plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('CurrExp/s009.pdf')
plt.clf()

#Top-K s009 Plot NLOS vs LOS


curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curveLOS=curveLOS/ curve[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curve[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Anti-Curriculum LOS',linestyle='--',color="orangered")
plt.plot(range(0, 256), curveNLOS, label=r'Anti-Curriculum NLOS',linestyle='-.',color="orangered")
curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_VANILLA.h5', 'MIXTURE')
curveLOS=curveLOS/ curve[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curve[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Standard LOS',linestyle='--',color="khaki")
plt.plot(range(0, 256), curveNLOS, label=r'Standard NLOS',linestyle='-.',color="khaki")
curveLOS, curveNLOS = plotNLOSvsLOS('CurrExp/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curveLOS=curveLOS/ curve[len(curveLOS) - 1]
curveNLOS=curveNLOS/ curve[len(curveNLOS) - 1]
plt.plot(range(0, 256), curveLOS, label=r'Curriculum LOS',linestyle='--',color="yellowgreen")
plt.plot(range(0, 256), curveNLOS, label=r'Curriculum NLOS',linestyle='-.',color="yellowgreen")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('CurrExp/s009.pdf')
plt.clf()


#Top-K s010 Plot

curve = plots010('CurrExp/MIXTURE_BETA_8_ANTI.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Anti-Curriculum',linestyle='--',color="orangered")
curve = plots010('CurrExp/MIXTURE_BETA_8_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Standard',linestyle='--',color="khaki")
curve = plots010('CurrExp/MIXTURE_BETA_8_CURR.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Curriculum',linestyle='--',color="yellowgreen")

plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s010')
plt.ylabel('top-K')
plt.grid()
plt.savefig('CurrExp/s010.pdf')
