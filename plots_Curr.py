import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.io
import pickle


plt.rc('font', family='serif', serif='Computer Modern Roman', size=14) # size=14 refers to the numbers on the x, y axes.
plt.rc('text', usetex=True)

reps=10
history_curr = pickle.load(open('Final/HistoryMIXTURE_BETA_8_CURR', "rb"))
### Plot Validation Acc and Validation Loss
acc_curr = history_curr['val_categorical_accuracy']

max_epoch=int((len(acc_curr) + 1)/reps)
acc_curr=np.reshape(acc_curr,(-1,max_epoch))
epochs_curr = range(0, max_epoch)
val_loss_curr = history_curr['val_loss']
val_loss_curr=np.reshape(val_loss_curr,(-1,max_epoch))
t5_curr = history_curr['val_top_5_accuracy']
t5_curr=np.reshape(t5_curr,(-1,max_epoch))
t10_curr = history_curr['val_top_10_accuracy']
t10_curr=np.reshape(t10_curr,(-1,max_epoch))

history_anti = pickle.load(open('Final/HistoryMIXTURE_BETA_8_ANTI', "rb"))
### Plot Validation Acc and Validation Loss
acc_anti = history_anti['val_categorical_accuracy']
acc_anti=np.reshape(acc_anti,(-1,max_epoch))
epochs_anti = range(0, max_epoch)
val_loss_anti = history_anti['val_loss']
val_loss_anti=np.reshape(val_loss_anti,(-1,max_epoch))
t5_anti = history_anti['val_top_5_accuracy']
t5_anti=np.reshape(t5_anti,(-1,max_epoch))
t10_anti = history_anti['val_top_10_accuracy']
t10_anti=np.reshape(t10_anti,(-1,max_epoch))

history_van = pickle.load(open('Final/HistoryMIXTURE_BETA_8_VANILLA', "rb"))
### Plot Validation Acc and Validation Loss
acc_van = history_van['val_categorical_accuracy']
acc_van=np.reshape(acc_van,(-1,max_epoch))
epochs_van = range(0, max_epoch)
val_loss_van = history_van['val_loss']
val_loss_van=np.reshape(val_loss_van,(-1,max_epoch))
t5_van = history_van['val_top_5_accuracy']
t5_van=np.reshape(t5_van,(-1,max_epoch))
t10_van = history_van['val_top_10_accuracy']
t10_van=np.reshape(t10_van,(-1,max_epoch))
epochs = range(0, max_epoch)

plt.plot(epochs, np.mean(acc_curr,axis=0),  linewidth=2)
std=np.sqrt(np.mean((acc_curr-np.mean(acc_curr,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(acc_curr,axis=0)-std, np.mean(acc_curr,axis=0)+std, color='tab:blue', alpha=.5)

plt.plot(epochs, np.mean(acc_van,axis=0), linewidth=2)
std=np.sqrt(np.mean((acc_van-np.mean(acc_van,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(acc_van,axis=0)-std, np.mean(acc_van,axis=0)+std,color='tab:orange', alpha=.5)

plt.plot(epochs, np.mean(acc_anti,axis=0), linewidth=2)
std=np.sqrt(np.mean((acc_anti-np.mean(acc_anti,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(acc_anti,axis=0)-std, np.mean(acc_anti,axis=0)+std,color='tab:green', alpha=.5)

plt.legend(['Curriculum', 'Standard', 'Anticurriculum'], fontsize=16, loc='lower right')
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Top-1 Accuracy", fontsize=16)

plt.grid()
axes = plt.gca()

plt.savefig("Top1Curves.pdf")
plt.clf()


plt.plot(epochs, np.mean(t5_curr,axis=0),  linewidth=2)
std=np.sqrt(np.mean((t5_curr-np.mean(t5_curr,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t5_curr,axis=0)-std, np.mean(t5_curr,axis=0)+std, color='tab:blue', alpha=.5)

plt.plot(epochs, np.mean(t5_van,axis=0), linewidth=2)
std=np.sqrt(np.mean((t5_van-np.mean(t5_van,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t5_van,axis=0)-std, np.mean(t5_van,axis=0)+std,color='tab:orange', alpha=.5)

plt.plot(epochs, np.mean(t5_anti,axis=0), linewidth=2)
std=np.sqrt(np.mean((t5_anti-np.mean(t5_anti,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t5_anti,axis=0)-std, np.mean(t5_anti,axis=0)+std,color='tab:green', alpha=.5)

plt.legend(['Curriculum', 'Standard', 'Anticurriculum'], fontsize=16, loc='lower right')
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Top-5 Accuracy", fontsize=16)

plt.grid()
axes = plt.gca()

plt.savefig("Top5Curves.pdf")
plt.clf()


plt.plot(epochs, np.mean(t10_curr,axis=0),  linewidth=2)
std=np.sqrt(np.mean((t10_curr-np.mean(t10_curr,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t10_curr,axis=0)-std, np.mean(t10_curr,axis=0)+std, color='tab:blue', alpha=.5)

plt.plot(epochs, np.mean(t10_van,axis=0), linewidth=2)
std=np.sqrt(np.mean((t10_van-np.mean(t10_van,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t10_van,axis=0)-std, np.mean(t10_van,axis=0)+std,color='tab:orange', alpha=.5)

plt.plot(epochs, np.mean(t10_anti,axis=0), linewidth=2)
std=np.sqrt(np.mean((t10_anti-np.mean(t10_anti,axis=0))**2,axis=0))
plt.fill_between(epochs, np.mean(t10_anti,axis=0)-std, np.mean(t10_anti,axis=0)+std,color='tab:green', alpha=.5)

plt.legend(['Curriculum', 'Standard', 'Anticurriculum'], fontsize=16, loc='lower right')
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Top-10 Accuracy", fontsize=16)

plt.grid()
axes = plt.gca()

plt.savefig("Top10Curves.pdf")
plt.clf()

TwoInOne=True


if(not TwoInOne):
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

    plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1,markersize=4)
    plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1, markersize=4)
    plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4)
    # Grid, legend, axis labels. Use LaTeX math font if necessary.
    plt.grid()
    plt.legend(['Curriculum', 'Standard', 'Anticurriculum'], fontsize=16, loc='lower right')
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
    plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1, markersize=4)
    plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4)
    plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4)# Grid, legend, axis labels. Use LaTeX math font if necessary.
    plt.grid()
    plt.legend(['Curriculum', 'Standard', 'Anticurriculum'], fontsize=16, loc='lower right')
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
else:

    curve = np.load('Final/CurvesMIXTURE_BETA_8_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_8_CURR.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_8_ANTI.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3 = np.mean(curve, axis=0)

    plt.rc('font', family='serif', serif='Computer Modern Roman', size=14)
    plt.rc('text', usetex=True)


    fig, axs = plt.subplots(2)
    fig.set_figheight(8)
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



