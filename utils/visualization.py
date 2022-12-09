from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_region(X, y, classifier, resolution=0.02, xlabel='x', ylabel='y'):
    markers = ('s', 'o', '^', 'x', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    samples = np.array([xx1.ravel(), xx2.ravel()]).T
    
    classifier.fit(X, y)
    Z = classifier.predict(samples).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    
    for i, c in enumerate(np.unique(y)):
        plt.scatter(
            X[y == c, 0],
            X[y == c, 1],
            alpha=0.8,
            c=cmap.colors[i],
            marker=markers[i],
            label=c,
            edgecolors='k'
        )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()