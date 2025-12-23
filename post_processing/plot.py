import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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
        for ax in [ax1, ax2]:
            ax.axvline(+cutoff, color='k', linestyle='--', linewidth=1, label='Cutoff')
            ax.axvline(-cutoff, color='k', linestyle='--', linewidth=1)
        ax1.legend()
        ax2.legend()
        
    plt.tight_layout()
    return fig, (ax1, ax2)

def keyrate_comparison_plot(
    results: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot key rate comparison similar to the reference image.
    
    Args:
        results (dict): Dictionary containing results at different stages with keys:
            - 'raw': {'beta_I_AB': float, 'I_E': float, 'KR': float}
            - 'alice_ps': {'beta_I_AB': float, 'I_E': float, 'KR': float}
            - 'bob_ps': {'beta_I_AB': float, 'I_E': float, 'KR': float}
        
    Returns:
        tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    stages = ['Raw\nPS', "Alice's\nPS", "Bob's\nPS"]
    stage_keys = ['raw', 'alice_ps', 'bob_ps']
    
    beta_I_AB = [results[key]['beta_I_AB'] for key in stage_keys]
    I_E = [results[key]['I_E'] for key in stage_keys]
    KR = [results[key]['KR'] for key in stage_keys]
    
    # Set up bar positions
    x = np.arange(len(stages))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, beta_I_AB, width, label=r'$\beta I_{AB}$', color='#7FB3D5', edgecolor='black')
    bars2 = ax.bar(x, I_E, width, label=r'$I_E$', color='#F0C19A', edgecolor='black')
    bars3 = ax.bar(x + width, KR, width, label='KR', color='#C5A8D9', edgecolor='black')
    
    optimal_value = np.max(KR)
    
    # Add optimal reference line
    ax.axhline(y=optimal_value, color='black', linestyle='--', linewidth=1.5, label='Optimal GG02')
    
    # Formatting
    ax.set_ylabel('Mutual Information', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add secondary y-axis for Key Rate
    ax2 = ax.twinx()
    ax2.set_ylabel('Key Rate (bits/symbols)', fontsize=12)
    ax2.set_yscale('log')
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax

