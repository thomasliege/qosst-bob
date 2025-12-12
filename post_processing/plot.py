import numpy as np
import matplotlib.pyplot as plt

def heatmap(
    data_x: np.ndarray,
    data_y: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D histogram (constellation diagram).

    Args:
        data_x (np.ndarray): X-axis data (I).
        data_y (np.ndarray): Y-axis data (Q).
        x_label (str): Label for X-axis.
        y_label (str): Label for Y-axis.
        title (str): Plot title.

    Returns:
        tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects.
    """
    fig = plt.figure()
    axes = fig.add_subplot()
    current_heatmap, xedges, yedges = np.histogram2d(data_x, data_y, bins=50)
    extent = (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1]))
    axes_image = axes.imshow(current_heatmap.T, extent=extent, origin="lower", cmap="rainbow")
    fig.colorbar(axes_image)
    axes.set_xlabel("I")
    axes.set_ylabel("Q")
    axes.grid()
    return fig, axes