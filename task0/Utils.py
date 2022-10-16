import numpy as np
import matplotlib.pyplot as plt
    
def Area(curve):
    x = curve.x()
    y = curve.y()
    if x.size < 2:
        return 0
    return np.sum((y[1:] + y[:-1]) / 2 * (x[1:] - x[:-1]))
        
def Plot(curve, ax=None, lim=[-0.01, 1.01], grid=True, format=".-"):
    x = curve.x()
    y = curve.y()
    xlabel = curve.xlabel
    ylabel = curve.ylabel
    title = curve.title
    if (ax is None):
        plt.plot(x, y, format)
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(grid)
    else:
        ax.plot(x, y, format)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(grid)
        