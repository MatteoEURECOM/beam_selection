from utils import plotNLOSvsLOS,readHistory,plotS010,plots009
import matplotlib.pyplot as plt

Net = 'IPC'
BETA=[0.2,0.4,0.6,0.8,1]

for beta in  BETA:
    curve_LOS,curve_NLOS=plotNLOSvsLOS('./saved_models/'+Net+'_BETA_'+str(int(beta*10))+'.h5',Net)
    plt.plot(range(0, 256), curve_NLOS / curve_NLOS[len(curve_NLOS) - 1], color = [1,beta,0.5*beta],lineStyle='--', label='NLoS '+Net+r' $\beta $='+str(beta), linewidth=1.5)
    plt.plot(range(0, 256), curve_LOS / curve_LOS[len(curve_LOS) - 1], color = [1,beta,0.5*beta], label='LoS '+Net+r' $\beta $='+str(beta), linewidth=1.5)
plt.xlabel('K')
plt.ylabel('top-K Accuracy')
plt.ylim((0.5, 1))
plt.xlim((0, 50))
plt.grid()
plt.legend()
plt.savefig(Net+'NLoS_LoS.png', dpi=150)
plt.show()


for beta in BETA:
    t=plotS010('./saved_models/PREDS_'+Net+'_BETA_'+str(int(beta*10))+'.csv')
    plt.plot(range(1, 257), plotS010('./saved_models/PREDS_'+Net+'_BETA_'+str(int(beta*10))+'.csv'),lineStyle='--', color = [1,beta,0.25*beta], label=Net+r' $\beta $='+str(beta), linewidth=1.5)
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.ylabel('top-K')
plt.title(Net+' top-K on s010')
plt.grid()
plt.savefig(Net+'s010.png', dpi=150)
plt.show()


for beta in BETA:
    curve=plots009('./saved_models/'+Net+'_BETA_'+str(int(beta*10))+'.h5',Net)
    plt.plot(range(0, 256), curve / curve[len(curve) - 1], color = [1,beta,0.25*beta],lineStyle='--', label='NLoS '+Net+r' $\beta $='+str(beta), linewidth=1.5)
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.ylabel('top-K')
plt.title(Net+' top-K on s009')
plt.grid()
plt.savefig(Net+'s009.png', dpi=150)
plt.show()


