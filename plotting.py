import numpy as np
import matplotlib
from matplotlib import pyplot as plt


plt.rc('font', family='serif', serif='Computer Modern Roman', size=14)
plt.rc('text', usetex=True)
fig, axs = plt.subplots(2)
fig.set_figheight(7)

curve = np.load('Final/CurvesMIXTURE_BETA_8_VANILLA.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_1 = np.mean(curve, axis=0)

curve = np.load('Final/CurvesMIXTURE_BETA_8_CURR.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_2 = np.mean(curve, axis=0)

curve = np.load('Final/CurvesMIXTURE_BETA_8_ANTI.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_3 = np.mean(curve, axis=0)


axs[0].plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1, markersize=4)
axs[0].plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
axs[0].plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4)
axs[0].grid()
axs[0].legend(['Curriculum','Standard' , 'Anticurriculum'], fontsize=14, loc='lower right')
axs[0].set_xlabel("$k$", fontsize=14)
axs[0].set_ylabel("Top-$k$ Accuracy", fontsize=14)
axs[0].set_xlim([0, 30])

curve = np.load('Final/CurvesTHMIXTURE_BETA_8_VANILLA.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_1 = np.mean(curve, axis=0)

curve = np.load('Final/CurvesTHMIXTURE_BETA_8_CURR.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_2 = np.mean(curve, axis=0)

curve = np.load('Final/CurvesTHMIXTURE_BETA_8_ANTI.npy')
curve = curve / curve[0, curve.shape[1] - 1]
data_y_3 = np.mean(curve, axis=0)

axs[1].plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1, markersize=4)
axs[1].plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1, markersize=4)
axs[1].plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1, markersize=4)
axs[1].grid()
axs[1].legend(['Curriculum','Standard', 'Anticurriculum'], fontsize=14, loc='lower right')
axs[1].set_xlabel("$k$", fontsize=14)
axs[1].set_ylabel("Top-$k$ Throughput Ratio", fontsize=14)
axs[1].set_xlim([0, 30])
plt.savefig("Curr2in1.pdf")