#!/bin/python
import numpy as np
import matplotlib.pyplot as plt

def plot_hexagon(A, title=None, block=True):
    X, Y = np.meshgrid(range(A.shape[0]), range(A.shape[-1]))
    X, Y = X*2, Y*2

    # Turn this into a hexagonal grid
    for i, k in enumerate(X):
        if i % 2 == 1:
            X[i] += 1
            Y[:,i] += 1

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    fig.set_dpi(600)
    im = ax.hexbin(
        X.reshape(-1),
        Y.reshape(-1),
        C=A.reshape(-1),
        gridsize=int(A.shape[0]/2)
    )

    # the rest of the code is adjustable for best output
    ax.set_aspect(1)
    ax.set(xlim=(-4, X.max()+4,), ylim=(-4, Y.max()+4))
    ax.axis(False)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    #plt.colorbar(im, shrink=0.7)

    if(title is not None):
        pass
        #plt.title(title)
    plt.savefig(f"{title}.png")
