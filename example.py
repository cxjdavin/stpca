from stpca import *

# Source: https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def show_example(W, X, strengths, save=False):
    # Create figure
    # [left, bottom, width, height], in fractions of the figure
    fig = plt.figure(figsize=(10,6))

    # Display color map
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax_cm = fig.add_axes([0.1, 0.85, 0.8, 0.05])
    ax_cm.set_axis_off()
    ax_cm.imshow(gradient, aspect='auto', cmap=cm.coolwarm)
    fig.text(0.05, 0.9, "min", va='center', ha='center', fontsize=20)
    fig.text(0.05, 0.85, "value", va='center', ha='center', fontsize=20)
    fig.text(0.95, 0.9, "max", va='center', ha='center', fontsize=20)
    fig.text(0.95, 0.85, "value", va='center', ha='center', fontsize=20)

    # Show planted signals
    ax_signal = fig.add_axes([0.05, 0.15, 0.4, 0.6])
    ax_signal.matshow(X, cmap=cm.coolwarm)
    fig.text(0.25, 0.075, "Signal $X$", va='center', ha='center', fontsize=20)

    # Show noise matrix
    ax_noise = fig.add_axes([0.55, 0.15, 0.4, 0.6])
    ax_noise.matshow(W, cmap=cm.coolwarm)
    fig.text(0.75, 0.075, "Noise $W$", va='center', ha='center', fontsize=20)

    if save:
        plt.savefig("example.png")
    #plt.show()

    # Generate combinations
    for L in range(1, 33):
        # Text for signals
        X_text = ""
        for i in range(len(strengths)):
            if i == 0:
                X_text += "{0} x_{1} x_{1}^\\top".format(strengths[i]*L, i+1)
            else:
                X_text += " + {0} x_{1} x_{1}^\\top".format(strengths[i]*L, i+1)
        X_text += "$"

        plt.matshow(W + L*X, cmap=cm.coolwarm)
        plt.title("$Y = W + " + X_text, fontsize=20)
        plt.axis('off')
        if save:
            if L < 10:
                plt.savefig("lambda0{0}.png".format(L))
            else:
                plt.savefig("lambda{0}.png".format(L))

def show_trace(Y, t, strengths, signals, recovered, maximizer_trace, filename_prefix, save=False):
    # Text for signals
    X_text = ""
    for i in range(len(strengths)):
        if i == 0:
            X_text += "{0} x_{1} x_{1}^\\top".format(strengths[i], i+1)
        else:
            X_text += " + {0} x_{1} x_{1}^\\top".format(strengths[i], i+1)
    X_text += "$"

    # Title
    title_text = "$Y = W + " + X_text + " with $t = {0}$".format(t)

    # Trace step-by-step
    # maximizer_trace is delimited by None between signals
    signal_num = 0
    step = 0
    while True:
        # Create figure
        # [left, bottom, width, height], in fractions of the figure
        fig = plt.figure(figsize=(10,6))
        fig.text(0.5, 0.9, title_text, va='center', ha='center', fontsize=20)
 
        # Show observation, overlaid with maximizer
        ax_signal = fig.add_axes([0.05, 0.15, 0.4, 0.6])
        u = maximizer_trace[step]
        if u is None:
            signal_num += 1
            step += 1
            if step == len(maximizer_trace):
                break
            else:
                u = maximizer_trace[step]
        supp = unsigned_supp(u)
        for i in supp:
            for j in supp:
                highlight_cell(j, i, ax=ax_signal, color="limegreen", linewidth=3)
        ax_signal.matshow(Y, cmap=cm.coolwarm)
        fig.text(0.25, 0.075, "Observation $Y$", va='center', ha='center', fontsize=20)

        # Use k to normalize colors in planted signals and recovered estimates
        k = len(unsigned_supp(signals[0]))
        mag = 1/np.sqrt(k)

        # Show signals
        for i in range(len(signals)):
            x = signals[i]
            ax_x = fig.add_axes([0.55, 0.65 - i*0.05, 0.361, 0.1])
            ax_x.set_axis_off()
            ax_x.matshow(np.matrix(x), cmap=cm.coolwarm, vmin=-mag, vmax=mag)
            fig.text(0.53, 0.7 - i*0.05, "$x_{0}$".format(i+1), va='center', ha='center', fontsize=15)

        # Show recovery, overlaid with recovered coordinates
        for i in range(len(recovered)):
            x = recovered[i]
            ax_x = fig.add_axes([0.55, 0.35 - i*0.05, 0.361, 0.1])
            ax_x.set_axis_off()
            ax_x.imshow(np.matrix(x), cmap=cm.coolwarm, vmin=-mag, vmax=mag)
            if signal_num == i:
                for j in set(supp).intersection(set(unsigned_supp(x))):
                    highlight_cell(j, 0, ax=ax_x, color="limegreen", linewidth=3)
            fig.text(0.53, 0.4 - i*0.05, "$\\widehat x_{0}$".format(i+1), va='center', ha='center', fontsize=15)

        if save:
            if L < 10:
                plt.savefig(filename_prefix + "0{0}.png".format(step - signal_num))
            else:
                plt.savefig(filename_prefix + "{0}.png".format(step - signal_num))
        #plt.show()
        step += 1

    # Save a copy without any highlights for static version
    # Create figure
    # [left, bottom, width, height], in fractions of the figure
    fig = plt.figure(figsize=(10,6))
    fig.text(0.5, 0.9, title_text, va='center', ha='center', fontsize=20)

    # Show observation
    ax_signal = fig.add_axes([0.05, 0.15, 0.4, 0.6])
    ax_signal.matshow(Y, cmap=cm.coolwarm)
    fig.text(0.25, 0.075, "Observation $Y$", va='center', ha='center', fontsize=20)

    # Use k to normalize colors in planted signals and recovered estimates
    k = len(unsigned_supp(signals[0]))
    mag = 1/np.sqrt(k)

    # Show signals
    for i in range(len(signals)):
        x = signals[i]
        ax_x = fig.add_axes([0.55, 0.65 - i*0.05, 0.361, 0.1])
        ax_x.set_axis_off()
        ax_x.matshow(np.matrix(x), cmap=cm.coolwarm, vmin=-mag, vmax=mag)
        fig.text(0.53, 0.7 - i*0.05, "$x_{0}$".format(i+1), va='center', ha='center', fontsize=15)

    # Show recovery, overlaid with recovered coordinates
    for i in range(len(recovered)):
        x = recovered[i]
        ax_x = fig.add_axes([0.55, 0.35 - i*0.05, 0.361, 0.1])
        ax_x.set_axis_off()
        ax_x.imshow(np.matrix(x), cmap=cm.coolwarm, vmin=-mag, vmax=mag)
        fig.text(0.53, 0.4 - i*0.05, "$\\widehat x_{0}$".format(i+1), va='center', ha='center', fontsize=15)

    if save:
        plt.savefig(filename_prefix + "static.png")
    #plt.show()

'''
Convering to animated GIF
convert -delay 25 lambda*.png lambda.gif

Converting to mp4 (see: https://trac.ffmpeg.org/wiki/Slideshow)
ffmpeg -framerate 3 -i lambda%02d.png -c:v libx264 -r 30 -pix_fmt yuv420p lambda.mp4
'''
if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    #p = 2
    #k = 5
    #n = 20
    #r = 1
    p = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    r = int(sys.argv[4])
    L = int(sys.argv[5])
    t = int(sys.argv[6])
    print("Parameters: p = {0}, k = {1}, n = {2}, r = {3}, L = {4}, t = {5}".format(p,k,n,r,L,t))

    # Generate instance
    strengths, signals, X, W = generate(p,k,n,r)
    strengths = [round(i,2) for i in strengths]

    # 1) Show X and W separately
    # 2) Animation
    # 3) Choose 3 L's to display
    #show_example(W, X, strengths, save=True)

    # Trace
    Y = W + L*X
    recovered, maximizer_trace = solve(Y, L, p, t, k, n, r)
    filename_prefix = "trace-L{0}-r{1}-t{2}-".format(L,r,t)
    show_trace(Y, t, [L*i for i in strengths], signals, recovered, maximizer_trace, filename_prefix, save=True)

