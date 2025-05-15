"""
Utility functions for data visualization
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from dagster import op, In, Out
from dagster import Any as dg_Any
from .image_processing_utils import compute_ndvi
from .metrics import compute_f1_scores
from .scaling_utils import min_max_scaling
from ..resources import ScalarTypeT


@op(out=Out(dg_Any))
def create_ndvi_colormap() -> LinearSegmentedColormap:
    """
    Returns a linearly segmented colormap for plotting an NDVI raster.
    """
    colors = ["red", "yellow", "lightgreen", "darkgreen"]
    ndvi_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    return ndvi_cmap


@op(
    ins={
        "data": In(dg_Any),
        "ground_truth_mask": In(dg_Any),
        "predicted_mask": In(dg_Any),
        "basename": In(str),
        "red_index": In(int),
        "green_index": In(int),
        "blue_index": In(int),
        "nir_index": In(int),
        "accuracy": In(dg_Any),
        "precision": In(dg_Any),
        "recall": In(dg_Any),
        "iou_value": In(dg_Any),
    }
)
def visualize_and_save_model_predictions(
    data: npt.NDArray[ScalarTypeT],
    ground_truth_mask: npt.NDArray[ScalarTypeT],
    predicted_mask: npt.NDArray[ScalarTypeT],
    basename: str = "image",
    red_index: int = 0,
    green_index: int = 1,
    blue_index: int = 2,
    nir_index: int = 3,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    iou_value: float | None = None,
) -> None:
    """
    Produce a 3 x 2 grid of subplots and save the visualization to disk.

    Top left subplot (0, 0): RGB image
    Top right subplot (0, 1): NDVI image
    Middle left subplot (1, 0): Ground truth binary mask
    Middle right subplot (1, 1): Ground truth mask applied on RGB image
    Bottom left subplot (2, 0): Predicted binary mask
    Bottom right subplot (2, 1): Predicted mask applied on RGB image

    Args:
        data: Data array with shape (img_height, img_width, n_bands)
        ground_truth_mask: Ground truth binary mask, shape = (img_height, img_width)
        predicted_mask: Predicted binary mask by a model, shape = (img_height, img_width)
        basename: Basename (including path) for image
        red_index: Index for red color channel in data array
        green_index: Index for green color channel in data array
        blue_index: Index for blue color channel in data array
        nir_index: Index for NIR channel in data array
        accuracy: Accuracy between predicted and ground truth masks
        precision: Precision between predicted and ground truth masks
        recall: Recall between predicted and ground truth masks
        iou_value: IoU between predicted and grounnd truth masks
    """
    if data.ndim != 3:
        raise ValueError("Input data array must be 3-dimensional.")
    img_height, img_width, n_bands = data.shape

    if red_index >= n_bands:
        raise ValueError(f"Index for red channel must be less than {n_bands}.")

    if blue_index >= n_bands:
        raise ValueError(f"Index for blue channel must be less than {n_bands}.")

    if green_index >= n_bands:
        raise ValueError(f"Index for green channel must be less than {n_bands}.")

    if nir_index >= n_bands:
        raise ValueError(f"Index for NIR channel must be less than {n_bands}.")

    if ground_truth_mask.shape != (img_height, img_width):
        raise ValueError(
            "Ground truth mask does not have the same image dimensions as the data."
        )

    if predicted_mask.shape != (img_height, img_width):
        raise ValueError(
            "Predicted mask does not have the same image dimensions as the data."
        )

    # Enhance color images through min-max scaling.
    data = min_max_scaling(data)

    # Extract RGB image from multispectral data array.
    rgb_img = data[:, :, [red_index, green_index, blue_index]]

    # Compute NDVI raster.
    ndvi = compute_ndvi(data[:, :, nir_index], data[:, :, red_index])
    ndvi_cmap = create_ndvi_colormap()

    # Plotting
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(9, 9),
        sharex=True,
        sharey=True,
    )
    # Top left subplot (0, 0): RGB image
    axes[0, 0].imshow(rgb_img, aspect="auto", interpolation="None")
    axes[0, 0].set_title("RGB image", fontsize=12)
    # Top right subplot (0, 1): NDVI image
    im = axes[0, 1].imshow(ndvi, aspect="auto", interpolation="None", cmap=ndvi_cmap)
    axes[0, 1].set_title("NDVI image", fontsize=12)
    # Middle left subplot (1, 0): Ground truth mask
    axes[1, 0].imshow(
        ground_truth_mask, aspect="auto", interpolation="None", cmap="Greens"
    )
    axes[1, 0].set_title("Ground truth mask", fontsize=12)
    # Middle right subplot (1, 1): Ground truth mask overlaid on NDVI image
    axes[1, 1].imshow(ndvi, aspect="auto", interpolation="None", cmap=ndvi_cmap)
    axes[1, 1].contour(ground_truth_mask, levels=[0.5], colors="purple", linewidths=0.8)
    axes[1, 1].set_title("Ground truth mask overlaid on NDVI image", fontsize=12)
    # Bottom left subplot (2, 0): Predicted mask
    axes[2, 0].imshow(
        predicted_mask, aspect="auto", interpolation="None", cmap="Greens"
    )
    axes[2, 0].set_title("Predicted mask", fontsize=12)
    # Bottom right subplot (2, 1): Predicted mask overlaid on NDVI image
    axes[2, 1].imshow(ndvi, aspect="auto", interpolation="None", cmap=ndvi_cmap)
    axes[2, 1].contour(predicted_mask, levels=[0.5], colors="purple", linewidths=0.8)
    axes[2, 1].set_title("Predicted mask overlaid on NDVI image", fontsize=12)
    # Remove ticks but keep labels
    for ax in axes.flat:
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    # Create space for adding colorbar.
    fig.subplots_adjust(
        top=0.96, bottom=0.05, left=0.01, right=0.89, wspace=0.1, hspace=0.1
    )

    # Set up colorbar for NDVI raster,
    cbar_ax = fig.add_axes((0.90, 0.10, 0.02, 0.80))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(np.arange(-1, 1.1, 0.2).tolist())
    cbar.set_label("NDVI", fontsize=12)

    # Add an annotation for results.
    results_annotation = ""
    if accuracy is not None:
        results_annotation += f"Accuracy = {100 * accuracy:.1f}%" + "   "
    if precision is not None:
        results_annotation += f"Precision = {100 * precision:.1f}%" + "   "
    if recall is not None:
        results_annotation += f"Recall = {100 * recall:.1f}%" + "   "
    if iou_value is not None:
        results_annotation += f"IoU = {100 * iou_value:.1f}%"
    fig.text(0.5, 0.02, results_annotation, ha="center", va="center", fontsize=12)

    # Save the figure to disk.
    plt.savefig(basename + "_results.png")
    plt.close()


@op(
    ins={
        "precision_vals": In(dg_Any),
        "recall_vals": In(dg_Any),
        "threshold_vals": In(dg_Any),
        "n_positive_samples": In(dg_Any),
        "n_samples": In(dg_Any),
        "basename": In(str),
    }
)
def plot_precision_recall_curve(
    precision_vals: npt.NDArray[ScalarTypeT],
    recall_vals: npt.NDArray[ScalarTypeT],
    threshold_vals: npt.NDArray[ScalarTypeT],
    n_positive_samples: int | None = None,
    n_samples: int | None = None,
    basename: str = "result",
) -> None:
    """
    Plot a precision-recall curve with color-coded binarization threshold values.

    Args:
        precision_vals: 1D array of precision values at different thresholds
        recall_vals: 1D array of recall values at different thresholds
        threshold_vals: 1D array of threshold values
        n_positive_samples: Count of positive samples in dataset
        n_total_samples: Total number of samples in dataset
        basename: Output plot basename (include path)
    """
    if not len(precision_vals) == len(recall_vals) == len(threshold_vals):
        raise ValueError(
            "Precision, recall, and threshold arrays must have the same length."
        )

    # F1 scores
    f1_scores = compute_f1_scores(precision_vals, recall_vals)
    max_f1_idx = np.argmax(f1_scores)

    # Area under precision-recall curve
    pr_auc = abs(np.trapz(precision_vals, recall_vals))

    # Prepare segments for color-coding
    points = np.array([recall_vals, precision_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize threshold values for color mapping.
    norm = plt.Normalize(vmin=0.0, vmax=1.0, clip=True)
    lc = LineCollection(segments, cmap="cividis", norm=norm)
    lc.set_array(threshold_vals)
    lc.set_linewidth(2)

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    # Add dotted line for baseline model.
    if n_samples is not None and n_positive_samples is not None:
        baseline_precision = n_positive_samples / n_samples
        ax.hlines(y=baseline_precision, xmin=0, xmax=1, linestyle=":", color="k")
        ax.annotate(
            "Baseline always positive model",
            xy=(0.2, baseline_precision + 0.01),
            xycoords="data",
            fontsize=12,
        )
    # Create colored line for precision-recall curve.
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label="Threshold")
    # Add annotation for area under curve computed using trapezoidal rule.
    ax.annotate(
        f"AUC = {pr_auc:.3f}",
        xy=(0.3, 0.8),
        xycoords="data",
        fontsize=12,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )
    # Add marker for data point with the highest F1 score.
    ax.plot(
        recall_vals[max_f1_idx],
        precision_vals[max_f1_idx],
        marker="*",
        markeredgecolor="black",
        markerfacecolor="darkorange",
        markersize=12,
        linestyle="None",
    )
    # Set axes limits.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    # Set axes labels.
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    plt.tight_layout()
    # Save and close the figure.
    plt.savefig(basename + "_precision_recall_curve.png")
    plt.close()
