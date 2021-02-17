import numpy as np
import matplotlib
from matplotlib import pyplot as plt
TwoInOne=True

if(not TwoInOne):
    # Allows to use LaTeX default font
    matplotlib.use('TkAgg')
    plt.rc('font', family='serif', serif='Computer Modern Roman', size=14) # size=14 refers to the numbers on the x, y axes.
    plt.rc('text', usetex=True)


    curve=np.load('Final/CurvesMIXTURE_BETA_0_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_1=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesMIXTURE_BETA_2_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_2=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesMIXTURE_BETA_4_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_3=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesMIXTURE_BETA_6_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_4=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesMIXTURE_BETA_8_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_5=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesMIXTURE_BETA_10_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_6=np.mean(curve,axis=0)

    # If you need more curves, just copy the above lines

    # Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
    # Always plot better accuracy first (from the top to the bottom curve)
    # I set default markersize, as our plots will be quite close to each other
    plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4, color="red")
    plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1,markersize=4,color="orangered")
    plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4,color="orange")
    plt.plot(np.arange(1, 31), data_y_4[0:30], marker='<', linewidth=1, markevery=1,markersize=4, color="khaki")
    plt.plot(np.arange(1, 31), data_y_5[0:30], marker='>', linewidth=1, markevery=1,markersize=4, color="yellowgreen")
    plt.plot(np.arange(1, 31), data_y_6[0:30], marker='s', linewidth=1, markevery=1,markersize=4, color="green")
    # Grid, legend, axis labels. Use LaTeX math font if necessary.
    plt.grid()
    plt.legend([r'$\beta=0$', r'$\beta=0.2$', r'$\beta=0.4$', r'$\beta=0.6$', r'$\beta=0.8$', r'$\beta=1$'], fontsize=16, loc='lower right')
    plt.xlabel("$k$", fontsize=16)
    plt.ylabel("Top-$k$ accuracy", fontsize=16)

    # Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
    # ylim can be set to default, as Matplotlib adjusts it automatically
    plt.xlim([0, 30])

    # Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
    #plt.show()

    # For the paper itself, save your figure as PDF for vector format
    plt.savefig("KDAcc.pdf")
    plt.clf()
    # Otherwise use .jpg with dpi=300
    # plt.savefig("example_plot.jpg", dpi=300)


    curve=np.load('Final/CurvesTHMIXTURE_BETA_0_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_1=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesTHMIXTURE_BETA_2_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_2=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesTHMIXTURE_BETA_4_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_3=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesTHMIXTURE_BETA_6_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_4=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesTHMIXTURE_BETA_8_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_5=np.mean(curve,axis=0)

    curve=np.load('Final/CurvesTHMIXTURE_BETA_10_VANILLA.npy')
    curve=curve/ curve[0,curve.shape[1] - 1]
    data_y_6=np.mean(curve,axis=0)

    # Load your data
    data_x_1 = np.arange(1, 51)

    data_x_2 = np.arange(1, 51)

    data_x_3 = np.arange(1, 51)
    # If you need more curves, just copy the above lines

    # Generating curves with proper style, marker order: o, ^, v, <, >, s. Always use empty markers and solid lines
    # Always plot better accuracy first (from the top to the bottom curve)
    # I set default markersize, as our plots will be quite close to each other
    plt.plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1,markersize=4, color="red")
    plt.plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1,markersize=4,color="orangered")
    plt.plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1,markersize=4,color="orange")
    plt.plot(np.arange(1, 31), data_y_4[0:30], marker='<', linewidth=1, markevery=1,markersize=4, color="khaki")
    plt.plot(np.arange(1, 31), data_y_5[0:30], marker='>', linewidth=1, markevery=1,markersize=4, color="yellowgreen")
    plt.plot(np.arange(1, 31), data_y_6[0:30], marker='s', linewidth=1, markevery=1,markersize=4, color="green")
    # Grid, legend, axis labels. Use LaTeX math font if necessary.
    plt.grid()
    plt.legend([r'$\beta=0$', r'$\beta=0.2$', r'$\beta=0.4$', r'$\beta=0.6$', r'$\beta=0.8$', r'$\beta=1$'], fontsize=16, loc='lower right')
    plt.xlabel("$k$", fontsize=16)
    plt.ylabel("Throughput Ratio", fontsize=16)

    # Set right border to be equal to k you use (I think 50 is a reasonable value), to show that the plot continues
    # ylim can be set to default, as Matplotlib adjusts it automatically
    plt.xlim([0, 30])

    # Only for visualization, if you want to save figure, comment the line below and uncomment one of the "savefig" lines
    #plt.show()

    # For the paper itself, save your figure as PDF for vector format
    plt.savefig("KDTh.pdf")
    plt.clf()
    # Otherwise use .jpg with dpi=300
    # plt.savefig("example_plot.jpg", dpi=300)
else:
    curve = np.load('Final/CurvesMIXTURE_BETA_0_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_2_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_4_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_6_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_4 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_8_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_5 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesMIXTURE_BETA_10_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_6 = np.mean(curve, axis=0)

    plt.rc('font', family='serif', serif='Computer Modern Roman',
           size=14)  # size=14 refers to the numbers on the x, y axes.
    plt.rc('text', usetex=True)

    fig, axs = plt.subplots(2)
    fig.set_figheight(7)

    axs[0].plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1, markersize=4,color="red")
    axs[0].plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1, markersize=4,color="orangered")
    axs[0].plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1, markersize=4,color="orange")
    axs[0].plot(np.arange(1, 31), data_y_4[0:30], marker='o', linewidth=1, markevery=1, markersize=4, color="khaki")
    axs[0].plot(np.arange(1, 31), data_y_5[0:30], marker='^', linewidth=1, markevery=1, markersize=4, color="yellowgreen")
    axs[0].plot(np.arange(1, 31), data_y_6[0:30], marker='v', linewidth=1, markevery=1, markersize=4, color="green")
    axs[0].grid()
    axs[0].legend([r'$\beta=0$', r'$\beta=0.2$', r'$\beta=0.4$', r'$\beta=0.6$', r'$\beta=0.8$', r'$\beta=1$'], fontsize=14, loc='lower right')
    axs[0].set_xlabel("$k$", fontsize=14)
    axs[0].set_ylabel("Top-$k$ Accuracy", fontsize=14)
    axs[0].set_xlim([0, 30])

    curve = np.load('Final/CurvesTHMIXTURE_BETA_0_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_1 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTHMIXTURE_BETA_2_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_2 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTHMIXTURE_BETA_4_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_3 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTHMIXTURE_BETA_6_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_4 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTHMIXTURE_BETA_8_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_5 = np.mean(curve, axis=0)

    curve = np.load('Final/CurvesTHMIXTURE_BETA_10_VANILLA.npy')
    curve = curve / curve[0, curve.shape[1] - 1]
    data_y_6 = np.mean(curve, axis=0)
    axs[1].plot(np.arange(1, 31), data_y_1[0:30], marker='o', linewidth=1, markevery=1, markersize=4, color="red")
    axs[1].plot(np.arange(1, 31), data_y_2[0:30], marker='^', linewidth=1, markevery=1, markersize=4, color="orangered")
    axs[1].plot(np.arange(1, 31), data_y_3[0:30], marker='v', linewidth=1, markevery=1, markersize=4, color="orange")
    axs[1].plot(np.arange(1, 31), data_y_4[0:30], marker='o', linewidth=1, markevery=1, markersize=4, color="khaki")
    axs[1].plot(np.arange(1, 31), data_y_5[0:30], marker='^', linewidth=1, markevery=1, markersize=4,
                color="yellowgreen")
    axs[1].plot(np.arange(1, 31), data_y_6[0:30], marker='v', linewidth=1, markevery=1, markersize=4, color="green")
    axs[1].grid()
    axs[1].legend([r'$\beta=0$', r'$\beta=0.2$', r'$\beta=0.4$', r'$\beta=0.6$', r'$\beta=0.8$', r'$\beta=1$'],
                  fontsize=14, loc='lower right')
    axs[1].set_xlabel("$k$", fontsize=14)
    axs[1].set_ylabel("Top-$k$ Throughput Ratio", fontsize=14)
    # plt.title(' Throughput Ratio on s009', fontsize=12)
    axs[1].set_xlim([0, 30])
    # axs[1].savefig("FinalTh.pdf")
    plt.savefig("KD2in1.pdf")