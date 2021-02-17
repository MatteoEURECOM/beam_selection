import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.io

plt.rc('font', family='serif', serif='Computer Modern Roman', size=14) # size=14 refers to the numbers on the x, y axes.
plt.rc('text', usetex=True)


curve=np.load('Final/CurvesMIXTURE_BETA_8_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_1=np.mean(curve,axis=0)

curve=np.load('Final/CurvesMIXTURE_BETA_8_CURR.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_2=np.mean(curve,axis=0)

curve=np.load('Final/CurvesMIXTURE_BETA_8_ANTI.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_3=np.mean(curve,axis=0)



# If you need more curves, just copy the above lines

# Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
# Always plot better accuracy first (from the top to the bottom curve)
# I set default markersize, as our plots will be quite close to each other
plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1,markersize=4)
plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4)
# Grid, legend, axis labels. Use LaTeX math font if necessary.
plt.grid()
plt.legend(['Standard', 'Curriculum', 'Anticurriculum'], fontsize=16, loc='lower right')
plt.xlabel("$k$", fontsize=16)
plt.ylabel("Top-$k$ accuracy", fontsize=16)

# Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
# ylim can be set to default, as Matplotlib adjusts it automatically
plt.xlim([0, 30])

# Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
#plt.show()

# For the paper itself, save your figure as PDF for vector format
plt.savefig("CurrAcc.pdf")
plt.clf()
# Otherwise use .jpg with dpi=300
# plt.savefig("example_plot.jpg", dpi=300)


curve=np.load('Final/CurvesTHMIXTURE_BETA_8_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_1=np.mean(curve,axis=0)

curve=np.load('Final/CurvesTHMIXTURE_BETA_8_CURR.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_2=np.mean(curve,axis=0)

curve=np.load('Final/CurvesTHMIXTURE_BETA_8_ANTI.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_3=np.mean(curve,axis=0)



# If you need more curves, just copy the above lines

# Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
# Always plot better accuracy first (from the top to the bottom curve)
# I set default markersize, as our plots will be quite close to each other
plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1,markersize=4)
plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4)# Grid, legend, axis labels. Use LaTeX math font if necessary.
plt.grid()
plt.legend(['Standard', 'Curriculum', 'Anticurriculum'], fontsize=16, loc='lower right')
plt.xlabel("$k$", fontsize=16)
plt.ylabel("Throughput Ratio", fontsize=16)

# Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
# ylim can be set to default, as Matplotlib adjusts it automatically
plt.xlim([0, 30])

# Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
#plt.show()

# For the paper itself, save your figure as PDF for vector format
plt.savefig("CurrTh.pdf")
plt.clf()
# Otherwise use .jpg with dpi=300
# plt.savefig("example_plot.jpg", dpi=300)