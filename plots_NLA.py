import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# Allows to use LaTeX default font
matplotlib.use('TkAgg')
plt.rc('font', family='serif', serif='Computer Modern Roman', size=14) # size=14 refers to the numbers on the x, y axes.
plt.rc('text', usetex=True)


curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_1=np.mean(curve,axis=0)

curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA_GAUSS.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_2=np.mean(curve,axis=0)

curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA_EMB.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_3=np.mean(curve,axis=0)

curve=np.load('KDExp/CurvesMIXTURE_BETA_8_VANILLA.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_4=np.mean(curve,axis=0)

# Load your data
data_x_1 = np.arange(1, 51)

data_x_2 = np.arange(1, 51)

data_x_3 = np.arange(1, 51)
# If you need more curves, just copy the above lines

# Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
# Always plot better accuracy first (from the top to the bottom curve)
# I set default markersize, as our plots will be quite close to each other
plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_4[0:30], marker='<', linewidth=1, markevery=2,markersize=4)
# Grid, legend, axis labels. Use LaTeX math font if necessary.
plt.grid()
plt.legend(['NLA Dot', 'NLA Gauss', 'NLA Embedded','Standard'], fontsize=12, loc='lower right')
plt.xlabel("$k$", fontsize=12)
plt.ylabel("Top-$k$ accuracy", fontsize=12)
plt.title(' Top-K on s009', fontsize=12)

# Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
# ylim can be set to default, as Matplotlib adjusts it automatically
plt.xlim([0, 30])

# Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
#plt.show()

# For the paper itself, save your figure as PDF for vector format
plt.savefig("NLAs009.pdf")
plt.clf()
# Otherwise use .jpg with dpi=300
# plt.savefig("example_plot.jpg", dpi=300)


curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_1=np.mean(curve,axis=0)

curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA010_GAUSS.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_2=np.mean(curve,axis=0)

curve=np.load('NLAExp/CurvesNON_LOCAL_MIXTURE_BETA_8_VANILLA010_EMB.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_3=np.mean(curve,axis=0)

curve=np.load('KDExp/CurvesMIXTURE_BETA_8_VANILLA010.npy')
curve=curve/ curve[0,curve.shape[1] - 1]
data_y_4=np.mean(curve,axis=0)

# Load your data
data_x_1 = np.arange(1, 51)

data_x_2 = np.arange(1, 51)

data_x_3 = np.arange(1, 51)
# If you need more curves, just copy the above lines

# Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
# Always plot better accuracy first (from the top to the bottom curve)
# I set default markersize, as our plots will be quite close to each other
plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=2,markersize=4)
plt.plot(np.arange(1, 31), data_y_4[0:30], marker='<', linewidth=1, markevery=2,markersize=4)
# Grid, legend, axis labels. Use LaTeX math font if necessary.
plt.grid()
plt.legend(['NLA Dot', 'NLA Gauss', 'NLA Embedded','Standard'], fontsize=12, loc='lower right')
plt.xlabel("$k$", fontsize=12)
plt.ylabel("Top-$k$ accuracy", fontsize=12)
plt.title(' Top-K on s010', fontsize=12)

# Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
# ylim can be set to default, as Matplotlib adjusts it automatically
plt.xlim([0, 30])

# Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
#plt.show()

# For the paper itself, save your figure as PDF for vector format
plt.savefig("NLAs010.pdf")
plt.clf()
# Otherwise use .jpg with dpi=300
# plt.savefig("example_plot.jpg", dpi=300)