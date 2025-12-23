import numpy as np
import matplotlib.pyplot as plt

def heatmap(
    data_x: np.ndarray,
    data_y: np.ndarray,
    title: str = "Heatmap",
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
    axes.set_title(title)
    axes.grid()
    return fig, axes

def histogram_comparison(
    symbols: np.ndarray,
    symbols_alice_pp: np.ndarray,
    symbols_bob_pp: np.ndarray,
    cutoff = None,
    title = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot histograms before and after post-selection for comparison.

    Args:
        symbols (np.ndarray): Symbols before post-selection.
        symbols_pp (np.ndarray): Symbols after post-selection.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]: The matplotlib figure and axes objects.
    """
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # Histogram comparison for Real part
    ax1.hist(symbols.real, bins=500, alpha=0.5, label='Before Post-Selection', color='blue')
    ax1.hist(symbols_alice_pp.real, bins=500, alpha=0.5, label='Alice Post-Selection', color='red')
    ax1.hist(symbols_bob_pp.real, bins=500, alpha=0.5, label='Bob Post-Selection', color='green')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram comparison for Imaginary part
    ax2.hist(symbols.imag, bins=500, alpha=0.5, label='Before Post-Selection', color='blue')
    ax2.hist(symbols_alice_pp.imag, bins=500, alpha=0.5, label='Alice Post-Selection', color='red')
    ax2.hist(symbols_bob_pp.imag, bins=500, alpha=0.5, label='Bob Post-Selection', color='green')
    ax2.set_xlabel('P')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if cutoff is not None:
        for ax, c in [(ax1, cutoff[0]), (ax2, cutoff[1])]:
            ax.axvline(+c, color='k', linestyle='--', linewidth=1, label='Cutoff')
            ax.axvline(-c, color='k', linestyle='--', linewidth=1)
        ax1.legend()
        ax2.legend()
        
    plt.tight_layout()
    return fig, (ax1, ax2)