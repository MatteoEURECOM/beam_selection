import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import load_dataset
from models import MULTIMODAL,LIDAR,GPS,MULTIMODAL_OLD,MIXTURE,NON_LOCAL_MIXTURE
import tensorflow as tf
import keras
from utils import plots009,plots010,finalMCAvg

'''
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('NLAvsBASE/HistoryNON_LOCAL_MIXTURE_BETA_8_CURR',5)
plt.errorbar([1,2,3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0',linestyle='None', marker='.',capsize=2,color="red")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('CurrExp/HistoryMIXTURE_BETA_8_CURR',5)
plt.errorbar([1.2,2.2,3.2], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0.8',linestyle='None', marker='.',capsize=2,color="yellowgreen")


axes = plt.gca()
axes.set_yticks(np.arange(0.3, 1, 0.05))
axes.set_yticks(np.arange(0.3, 1, 0.01), minor=True)
axes.set_xticks([1.2,2.2,3.2])
axes.set_xticklabels(['Top-1','Top-5','Top-10'])
axes.grid(which='minor', alpha=0.2)
axes.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig('NLAvsBASE/Metrics.pdf')
plt.show()
plt.clf()


curve = plots009('NLAvsBASE/NON_LOCAL_MIXTURE_BETA_8_CURR.h5', 'NON_LOCAL_MIXTURE','ABSOLUTE','dot',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'NLA Mixture',linestyle='--',color="red")
curve = plots009('CurrExp/MIXTURE_BETA_8_CURR.h5', 'MIXTURE','ABSOLUTE','gaussian',2)
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'Mixture',linestyle='--',color="yellowgreen")

plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('NLAvsBASE/NLAvsStandard.pdf')
plt.show()
plt.clf()

'''
curve=np.load('KDExp/CurvesMIXTURE_BETA_0_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0', color="red")

curve=np.load('KDExp/CurvesMIXTURE_BETA_2_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0.2',color="orangered")

curve=np.load('KDExp/CurvesMIXTURE_BETA_4_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0.4',color="orange")

curve=np.load('KDExp/CurvesMIXTURE_BETA_6_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean, label=r'$\beta$=0.6', color="khaki")

curve=np.load('KDExp/CurvesMIXTURE_BETA_8_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean,  label=r'$\beta$=0.8', color="yellowgreen")


curve=np.load('KDExp/CurvesMIXTURE_BETA_10_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean, label=r'$\beta$=1', color="green")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('KDExp/s009.pdf')

plt.clf()


curve=np.load('KDExp/CurvesMIXTURE_BETA_0_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0', color="red")

curve=np.load('KDExp/CurvesMIXTURE_BETA_2_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0.2',color="orangered")

curve=np.load('KDExp/CurvesMIXTURE_BETA_4_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257),mean, label=r'$\beta$=0.4',color="orange")

curve=np.load('KDExp/CurvesMIXTURE_BETA_6_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean, label=r'$\beta$=0.6', color="khaki")

curve=np.load('KDExp/CurvesMIXTURE_BETA_8_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean,  label=r'$\beta$=0.8', color="yellowgreen")


curve=np.load('KDExp/CurvesMIXTURE_BETA_10_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
mean=np.mean(curve,axis=0)
std=np.sqrt(np.sum((curve-mean)**2,axis=0)/(curve.shape[0]-1))
plt.plot(np.arange(1,257), mean, label=r'$\beta$=1', color="green")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()

plt.savefig('KDExp/s010.pdf')

plt.clf()




#METRICS Plot

val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_0_VANILLA',5)
plt.errorbar([1,2,3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0',linestyle='None', marker='.',capsize=2,color="red")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_2_VANILLA',5)
plt.errorbar([1.1,2.1,3.1], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0.2',linestyle='None', marker='.',capsize=2,color="orangered")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_4_VANILLA',5)
plt.errorbar([1.2,2.2,3.2], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0.4',linestyle='None', marker='.',capsize=2,color="orange")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_6_VANILLA',5)
plt.errorbar([1.3,2.3,3.3], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0.6',linestyle='None', marker='.',capsize=2,color="khaki")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_8_VANILLA',5)
plt.errorbar([1.4,2.4,3.4], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=0.8',linestyle='None', marker='.',capsize=2,color="yellowgreen")
val_acc_mean,val_acc_std,t5_mean,t5_std,t10_mean,t10_std=finalMCAvg('KDExp/HistoryMIXTURE_BETA_10_VANILLA',5)
plt.errorbar([1.5,2.5,3.5], [val_acc_mean,t5_mean,t10_mean], [val_acc_std,t5_std,t10_std], label=r'$\beta$=1',linestyle='None', marker='.',capsize=2,color="green")

axes = plt.gca()
axes.set_yticks(np.arange(0.3, 1, 0.05))
axes.set_yticks(np.arange(0.3, 1, 0.01), minor=True)
axes.set_xticks([1.2,2.2,3.2])
axes.set_xticklabels(['Top-1','Top-5','Top-10'])
axes.grid(which='minor', alpha=0.2)
axes.grid(which='major', alpha=0.5)
plt.legend()
plt.savefig('KDExp/Metrics.pdf')
plt.clf()

#Top-K s009 Plot

curve = plots009('KDExp/MIXTURE_BETA_0_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0',linestyle='--',color="red")
curve = plots009('KDExp/MIXTURE_BETA_2_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.2',linestyle='--',color="orangered")
curve=plots009('KDExp/MIXTURE_BETA_4_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.4',linestyle='--',color="orange")
curve = plots009('KDExp/MIXTURE_BETA_6_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.6',linestyle='--',color="khaki")
curve = plots009('KDExp/MIXTURE_BETA_8_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.8',linestyle='--',color="yellowgreen")
curve = plots009('KDExp/MIXTURE_BETA_10_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=1',linestyle='--',color="green")
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s009')
plt.ylabel('top-K')
plt.grid()
plt.savefig('KDExp/s009.pdf')
plt.clf()

#Top-K s010 Plot

curve = plots010('KDExp/MIXTURE_BETA_0_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0',linestyle='--',color="red")
curve = plots010('KDExp/MIXTURE_BETA_2_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.2',linestyle='--',color="orangered")
curve = plots010('KDExp/MIXTURE_BETA_4_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.4',linestyle='--',color="orange")
curve = plots010('KDExp/MIXTURE_BETA_6_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.6',linestyle='--',color="khaki")
curve = plots010('KDExp/MIXTURE_BETA_8_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=0.8',linestyle='--',color="yellowgreen")
curve = plots010('KDExp/MIXTURE_BETA_10_FINAL_VANILLA.h5', 'MIXTURE')
curve=curve/ curve[len(curve) - 1]
plt.plot(range(0, 256), curve, label=r'$\beta$=1',linestyle='--',color="green")

plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.title(' Top-K on s010')
plt.ylabel('top-K')
plt.grid()
plt.savefig('KDExp/s010.pdf')
