"""
Matplot lib visualization helpers for tensor-anomaly outputs.

All functions return (fig, ax) so callers can either display or save the figure without side effects from this module.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

def plot_sensor_timeseries(
        tensor: np.ndarray,
        machine_idx: int = 0,
        title: str | None = None,
) -> tuple[Figure, Axes]:
    """Line plot of all sensor readings for a single machine over time.
    
    Parameters
    
    tensor:
        Array of shape (n_machines, n_sensors, n_timesteps).
    machine_idx:
        Which machine to plot.
    title:
        Optional figure title; defaults to "Machine {idx} - All Sensors".
    
    Returns
    
    (fig, ax)
    """

    _, n_sensors, n_timesteps = tensor.shape
    if title is None:
        title = f"Machine {machine_idx} - All Sensors"

    fig, ax = plt.subplots(figsize=(12,4))
    t = np.arange(n_timesteps)
    cmap = plt.get_cmap("tab10")

    for s in range(n_sensors):
        ax.plot(
            t,
            tensor[machine_idx, s, :],
            label = f"Sensor {s}",
            color = cmap( s % 10),
            linewidth = 1.2,
            alpha = 0.85
        )

    ax.set_xlabel("Timestep (hours)")
    ax.set_ylabel("Sensor Reading")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=min(n_sensors, 4), fontsize=8)
    fig.tight_layout()
    return fig, ax

def plot_anomaly_heatmap(
    scores: np.ndarray,
    machine_names: list[str] | None = None,
    title: str = "Anomaly Score Heatmap (machines x time)",
) -> tuple[Figure, Axes]:
    """Heat map of anomaly scores with machines on the y-axis and time on x.
    
    Parameters
    
    scores:
        Array of shape (n_machines, n_timestamps) as return by 
        AnomalyDetector.score().
    machine_names:
        Optional list of human-readable machine labels.  Defaults to ["machine_00", ...] 
    title:
        Optional figure title; defaults to "Anomaly Score Heatmap (machines x time)".

    Returns

    (fig, ax)   
    """

    n_machines, n_timesteps = scores.shape
    if machine_names is None:
        machine_names = [f"machine_{i:02d}" for i in range(n_machines)]

    fig, ax = plt.subplots(figsize=(14,max(4, n_machines * 0.35)))
    im = ax.imshow(
        scores,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Reconstruction error (normalized)", fontsize=9)

    ax.set_yticks(np.arange(n_machines))
    ax.set_yticklabels(machine_names, fontsize=8)
    ax.set_xlabel("Timestep (hours)")
    ax.set_title(title)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    fig.tight_layout()
    return fig, ax

def plot_reconstruction_error(
    normal_errors: np.ndarray,
    test_errors: np.ndarray,
    threshold: float,
    title: str = "Reconstruction Error Distribution: Normal vs Test",
) -> tuple[Figure, Axes]:
    """Histogram comparing training vs test set reconstruction errors.
    
    Parameters
    
    normal_errors: 
        1-D array of per-slice scoreson normal(training) data)
    test_errors:
        1-D array of per-slice scores on test data
    threshold: 
        Fitted anomaly threshold; drawn as a vertical dashed line
    title:
        Figure title
        
    Returns:
    
    (fig, ax)
    """

    fig, ax = plt.subplots(figsize=(9,5))

    bins = np.linspace(
        0,
        max(np.max(normal_errors), np.max(test_errors)) * 1.05,
        50,
    )

    ax.hist(
        normal_errors.ravel(),
        bins = bins,
        alpha = 0.6,
        color = "steelblue",
        label = "Normal (training)",
        density = True,
    )

    ax.hist(
        test_errors.ravel(),
        bins = bins,
        alpha = 0.6, 
        color = "tomato",
        label = "Test",
        density = True,
    )

    ax.axvline(
        threshold,
        color = "black",
        linestyle = "--",
        linewidth = 1.5,
        label = f"Threshold ({threshold:.4f})",
    )

    ax.set_xlabel("Reconstruction error (normalized)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax