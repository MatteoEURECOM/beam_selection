from utils import plotNLOSvsLOS,readHistory,plotS010,plots009
import matplotlib.pyplot as plt

CURRICULUM=False
Net = 'MIXTURE'
BETA=[0.2,0.4,0.6,0.8,1]
'''
for beta in  BETA:
    if(CURRICULUM):
        curve_LOS, curve_NLOS = plotNLOSvsLOS('./saved_models/' + Net + '_BETA_' + str(int(beta * 10)) + 'CURRICULUM.h5', Net)
    else:
        curve_LOS,curve_NLOS=plotNLOSvsLOS('./saved_models/'+Net+'_BETA_'+str(int(beta*10))+'.h5',Net)
    plt.plot(range(0, 256), curve_NLOS / curve_NLOS[len(curve_NLOS) - 1], color = [1,beta,0.5*beta],lineStyle='--', label='NLoS '+Net+r' $\beta $='+str(beta), linewidth=1.5)
    plt.plot(range(0, 256), curve_LOS / curve_LOS[len(curve_LOS) - 1], color = [1,beta,0.5*beta], label='LoS '+Net+r' $\beta $='+str(beta), linewidth=1.5)
plt.xlabel('K')
plt.ylabel('top-K Accuracy')
plt.ylim((0.5, 1))
plt.xlim((0, 50))
plt.grid()
plt.legend()
if(CURRICULUM):
    plt.savefig(Net + 'NLoS_LoS_CURRICULUM.png', dpi=150)
else:
    plt.savefig(Net + 'NLoS_LoS.png', dpi=150)

plt.show()
'''
acc_max=0
beta_max=0
for beta in BETA:
    if (CURRICULUM):
        t = plotS010('./saved_models/PREDS_' + Net + '_BETA_' + str(int(beta * 10)) + 'CURRICULUM.csv')
    else:
        t = plotS010('./saved_models/PREDS_' + Net + '_BETA_' + str(int(beta * 10)) + '.csv')
    if(acc_max<t[9]):
        acc_max=t[9]
        beta_max=beta
    plt.plot(range(1, 257), t,lineStyle='--', color = [1,beta,0.25*beta], label=Net+r' $\beta $='+str(beta), linewidth=1.5)
print('s010:')
print(beta_max)
print(acc_max)
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.ylabel('top-K')

plt.grid()
if(CURRICULUM):
    plt.title(Net + ' top-K on s010 with curriculum learning')
    plt.savefig(Net + 's010_CURRICULUM.png', dpi=150)
else:
    plt.title(Net + ' top-K on s010')
    plt.savefig(Net + 's010.png', dpi=150)

plt.show()

acc_max=0
beta_max=0
for beta in BETA:
    if (CURRICULUM):
        curve = plots009('./saved_models/' + Net + '_BETA_' + str(int(beta * 10)) + 'CURRICULUM.h5', Net)
        curve=curve/ curve[len(curve) - 1]
    else:
        curve = plots009('./saved_models/' + Net + '_BETA_' + str(int(beta * 10)) + '.h5', Net)
        curve=curve/ curve[len(curve) - 1]
    if (acc_max <curve[9]):
        acc_max = curve[9]
        beta_max = beta
    plt.plot(range(0, 256), curve , color = [1,beta,0.25*beta],lineStyle='--', label=Net+r' $\beta $='+str(beta), linewidth=1.5)
print('s009:')
print(beta_max)
print(acc_max)
plt.legend()
plt.ylim(0.5, 1)
plt.xlim(0, 50)
plt.xlabel('K')
plt.ylabel('top-K')

plt.grid()
if(CURRICULUM):
    plt.title(Net + ' top-K on s009 with curriculum learning')
    plt.savefig(Net + 's009_CURRICULUM.png', dpi=150)
else:
    plt.title(Net + ' top-K on s009')
    plt.savefig(Net + 's009.png', dpi=150)

plt.show()


