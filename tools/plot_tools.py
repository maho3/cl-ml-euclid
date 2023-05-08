
# Imports
import numpy as np
import matplotlib.pyplot as plt


def binned_plot(X, Y, n=10, percentiles=[0, 34],
                alpha=0.2, linealpha=0.7, ax=None, **kwargs):
    """
    Plots percentiles of Y at n bins evenly distributed along the X axis.

    Attributes
    ----------
    X,Y : array-like
        The horizontal/vertical coordinates of the data points
    n : int
        Number of bins with which to evenly partition the x axis
    percentiles : list of ints/floats
        Percentiles to measure and display of the Y data in each bin.
        A 0-th percentile corresponds to a solid line at the median
    alpha : float in [0,1]
        Alpha value for transparency of shaded percentile region
    linealpha : float in [0,1]
        Alpha value for transparency of borders of percentile regions
    ax : matplotlib.Axis
        Axis on which to generte images
    **kwargs : Line2D properties
        To be passed to matplotlib.pyplot.fill_between and
        matplotlib.pyplot.plot
    """
    # Calculation
    calc_percent = []
    for p in percentiles:
        if p == 0:
            calc_percent.append(50)
        elif p <= 50:
            calc_percent.append(50-p)
            calc_percent.append(50+p)
        else:
            raise Exception('Percentile > 50')

    bin_edges = np.linspace(X.min()*0.9999, X.max()*1.0001, n+1)

    dtype = [(str(i), 'f') for i in calc_percent]
    bin_data = np.zeros(shape=(n,), dtype=dtype)

    for i in range(n):
        y = Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])]

        if len(y) == 0:
            bin_data[i] = None
            continue

        y_p = np.percentile(y, calc_percent)

        bin_data[i] = tuple(y_p)

    # Plotting
    if ax is None:
        f, ax = plt.subplots()

    bin_centers = [np.mean(bin_edges[i:i+2]) for i in range(n)]
    for p in percentiles:
        if p == 0:
            ax.plot(bin_centers, bin_data['50'], alpha=linealpha, **kwargs)
        else:
            ax.fill_between(bin_centers,
                            bin_data[str(50-p)],
                            bin_data[str(50+p)],
                            alpha=alpha, linewidth=0,
                            **kwargs)
            ax.plot(bin_centers, bin_data[str(50-p)],
                    alpha=linealpha, linewidth=1, **kwargs)
            ax.plot(bin_centers, bin_data[str(50+p)],
                    alpha=linealpha, linewidth=1, **kwargs)

    return bin_data, bin_edges
