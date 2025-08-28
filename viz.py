import numpy as np, matplotlib.pyplot as plt

def plot_overlaid_traj(truth, pred, dt, labels=None, title="Learned vs Truth", phase=True):
    truth, pred = np.asarray(truth), np.asarray(pred)
    Tp1, nx = truth.shape; t = np.arange(Tp1)*dt
    if labels is None: labels = [f"x{i+1}" for i in range(nx)]
    plt.figure(figsize=(6, 2.4*nx))
    for i in range(nx):
        ax = plt.subplot(nx,1,i+1)
        ax.plot(t, truth[:,i], lw=2, label="truth")
        ax.plot(t, pred[:,i],  "--", lw=2, label="learned")
        ax.set_ylabel(labels[i]); ax.grid(True, ls="--", alpha=0.3)
        if i==0: ax.set_title(title)
    plt.xlabel("time"); plt.legend(); plt.tight_layout(); plt.show()
    if phase and nx==2:
        plt.figure(figsize=(5,5))
        plt.plot(truth[:,0], truth[:,1], lw=2, label="truth")
        plt.plot(pred[:,0],  pred[:,1],  "--", lw=2, label="learned")
        plt.legend(); plt.axis("equal"); plt.grid(True, ls="--", alpha=0.3)
        plt.title("Phase portrait"); plt.tight_layout(); plt.show()
