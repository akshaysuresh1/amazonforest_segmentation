"""
Unit tests for plotting utilities
"""

from typing import Tuple
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from amazon_seg_project.ops.plotting_utils import (
    create_ndvi_colormap,
    visualize_and_save_model_predictions,
    plot_precision_recall_curve,
)


def almost_equal_color(
    color1: Tuple[float, float, float], color2: Tuple[float, float, float], delta=0.05
) -> bool:
    "Check if RGB triplets for two colors are equal."
    return all(abs(a - b) < delta for a, b in zip(color1, color2))


def test_ndvi_colormap_creation_success() -> None:
    """
    Test for successful creation of colormap for NDVI raster.
    """
    cmap = create_ndvi_colormap()
    # Check type
    assert isinstance(cmap, LinearSegmentedColormap)
    # Check name
    assert cmap.name == "custom"
    # Check color interpolation at key points
    # Red at 0.0, Yellow at ~0.33, Lightgreen at ~0.66, Darkgreen at 1.0
    assert almost_equal_color(cmap(0.0)[:3], (1.0, 0.0, 0.0)), (
        "Red not matched to the start of colormap."
    )
    assert almost_equal_color(cmap(0.33)[:3], (1.0, 1.0, 0.0)), (
        "Yellow not matched to the color at 0.66 of the colormap."
    )
    assert almost_equal_color(cmap(0.66)[:3], (0.5647, 0.9333, 0.5647)), (
        "Light green not matched to the color at 0.66 of the colormap."
    )
    assert almost_equal_color(cmap(1.0)[:3], (0.0, 0.3921, 0.0)), (
        "Dark green not matched to the end of the colormap."
    )


def test_visualize_model_predictions_invalid_data_dim() -> None:
    """
    Check for correct raise of ValueError upon encountering data.dim != 3.
    """
    with pytest.raises(ValueError, match="Input data array must be 3-dimensional."):
        # Create fake inputs.
        data = np.random.randn(16, 16)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(data, ground_truth_mask, predicted_mask)


def test_visualize_model_predictions_invalid_red_index() -> None:
    """
    Ensure correct raise of ValueError when supplied red index is an invalid color channel in data.
    """
    n_bands = 3
    with pytest.raises(
        ValueError, match=f"Index for red channel must be less than {n_bands}."
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, n_bands)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(
            data, ground_truth_mask, predicted_mask, red_index=n_bands
        )


def test_visualize_model_predictions_invalid_blue_index() -> None:
    """
    Ensure correct raise of ValueError when supplied blue index is an invalid color channel in data.
    """
    n_bands = 3
    with pytest.raises(
        ValueError, match=f"Index for blue channel must be less than {n_bands}."
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, n_bands)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(
            data, ground_truth_mask, predicted_mask, blue_index=n_bands
        )


def test_visualize_model_predictions_invalid_green_index() -> None:
    """
    Ensure correct raise of ValueError when supplied green index is an invalid color channel in data.
    """
    n_bands = 3
    with pytest.raises(
        ValueError, match=f"Index for green channel must be less than {n_bands}."
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, n_bands)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(
            data, ground_truth_mask, predicted_mask, green_index=n_bands
        )


def test_visualize_model_predictions_invalid_nir_index() -> None:
    """
    Ensure correct raise of ValueError when supplied NIR index is an invalid color channel in data.
    """
    n_bands = 3
    with pytest.raises(
        ValueError, match=f"Index for NIR channel must be less than {n_bands}."
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, n_bands)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(
            data, ground_truth_mask, predicted_mask, nir_index=n_bands
        )


def test_visualize_model_predictions_invalid_grouth_truth_mask() -> None:
    """
    Ensure correct raise of ValueError when ground truth mask has invalid shape.
    """
    with pytest.raises(
        ValueError,
        match="Ground truth mask does not have the same image dimensions as the data.",
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, 4)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(32, 32))
        predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        visualize_and_save_model_predictions(data, ground_truth_mask, predicted_mask)


def test_visualize_model_predictions_invalid_predicted_mask() -> None:
    """
    Ensure correct raise of ValueError when predicted mask has invalid shape.
    """
    with pytest.raises(
        ValueError,
        match="Predicted mask does not have the same image dimensions as the data.",
    ):
        # Create fake inputs.
        data = np.random.randn(16, 16, 4)
        ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
        predicted_mask = np.random.randint(low=0, high=2, size=(32, 32))

        # Call the test function.
        visualize_and_save_model_predictions(data, ground_truth_mask, predicted_mask)


@patch("matplotlib.pyplot.close")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.subplots")
@patch("amazon_seg_project.ops.plotting_utils.create_ndvi_colormap")
@patch("amazon_seg_project.ops.plotting_utils.compute_ndvi")
@patch("amazon_seg_project.ops.plotting_utils.min_max_scaling")
def test_visualize_and_save_model_predictions(
    mock_minmax_scaling: MagicMock,
    mock_compute_ndvi: MagicMock,
    mock_create_ndvi_colormap: MagicMock,
    mock_plt_subplots: MagicMock,
    mock_plt_savefig: MagicMock,
    mock_plt_close: MagicMock,
) -> None:
    """
    Test for successful execution of visualize_and_save_model_predictions().
    """
    # Set up inputs.
    multispec_data = np.random.randn(16, 16, 4)
    ground_truth_mask = np.random.randint(low=0, high=2, size=(16, 16))
    predicted_mask = np.random.randint(low=0, high=2, size=(16, 16))
    basename = "test_plot"
    red_index, green_index, blue_index, nir_index = 0, 1, 2, 3
    accuracy, precision, recall, iou_score = 0.95, 0.96, 0.93, 0.91
    rgb_data = multispec_data[:, :, [red_index, green_index, blue_index]]

    # Expected annotation in figure
    results_annotation = f"Accuracy = {100 * accuracy:.1f}%   "
    results_annotation += f"Precision = {100 * precision:.1f}%   "
    results_annotation += f"Recall = {100 * recall:.1f}%   "
    results_annotation += f"IoU = {100 * iou_score:.1f}%"

    # Set up mock intermediates.
    mock_minmax_scaling.return_value = multispec_data
    mock_ndvi = mock_compute_ndvi.return_value
    mock_ndvi_cmap = mock_create_ndvi_colormap.return_value
    mock_fig = MagicMock(name="mock-figure")
    # Mock axes
    mock_axes = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            mock_axes[i, j] = MagicMock(name=f"axes-{i}-{j}")
    mock_plt_subplots.return_value = (mock_fig, mock_axes)
    mock_ndvi_im = MagicMock(name="mock-ndvi-im")
    mock_axes[0, 1].imshow.return_value = mock_ndvi_im
    # Mock colorbar axis
    mock_cbar_ax = MagicMock(name="mock-cbar-ax")
    mock_fig.add_axes.return_value = mock_cbar_ax
    # Mock colorbar object
    mock_cbar = MagicMock(name="mock-cbar")
    mock_fig.colorbar.return_value = mock_cbar

    # Call the test function.
    visualize_and_save_model_predictions(
        multispec_data,
        ground_truth_mask,
        predicted_mask,
        basename=basename,
        red_index=red_index,
        blue_index=blue_index,
        green_index=green_index,
        nir_index=nir_index,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        iou_value=iou_score,
    )

    # Assertions for data transformations
    mock_minmax_scaling.assert_called_once_with(multispec_data)
    ndvi_args, _ = mock_compute_ndvi.call_args

    # Compare arrays elementwise.
    np.testing.assert_array_equal(ndvi_args[0], multispec_data[:, :, nir_index])
    np.testing.assert_array_equal(ndvi_args[1], multispec_data[:, :, red_index])
    mock_create_ndvi_colormap.assert_called_once()

    # Assertions for plotting routines
    mock_plt_subplots.assert_called_once_with(
        nrows=3,
        ncols=2,
        figsize=(9, 9),
        sharex=True,
        sharey=True,
    )
    # Top left subplot (0, 0): RGB image
    ax00_args, ax00_kwargs = mock_axes[0, 0].imshow.call_args
    np.testing.assert_array_equal(ax00_args[0], rgb_data)
    assert ax00_kwargs["aspect"] == "auto"
    assert ax00_kwargs["interpolation"] == "None"
    mock_axes[0, 0].set_title.assert_called_once_with("RGB image", fontsize=12)

    # Top right subplot (0, 1): NDVI image
    mock_axes[0, 1].imshow.assert_called_once_with(
        mock_ndvi, aspect="auto", interpolation="None", cmap=mock_ndvi_cmap
    )
    mock_axes[0, 1].set_title.assert_called_once_with("NDVI image", fontsize=12)

    # Middle left subplot (1, 0): Ground truth mask
    ax10_args, ax10_kwargs = mock_axes[1, 0].imshow.call_args
    np.testing.assert_array_equal(ax10_args[0], ground_truth_mask)
    assert ax10_kwargs["aspect"] == "auto"
    assert ax10_kwargs["interpolation"] == "None"
    assert ax10_kwargs["cmap"] == "Greens"
    mock_axes[1, 0].set_title.assert_called_once_with("Ground truth mask", fontsize=12)

    # Middle right subplot (1, 1): Ground truth mask overlaid on NDVI image
    mock_axes[1, 1].imshow.assert_called_once_with(
        mock_ndvi, aspect="auto", interpolation="None", cmap=mock_ndvi_cmap
    )
    ax11_args, ax11_kwargs = mock_axes[1, 1].contour.call_args
    np.testing.assert_array_equal(ax11_args[0], ground_truth_mask)
    assert ax11_kwargs["levels"] == [0.5]
    assert ax11_kwargs["colors"] == "purple"
    assert ax11_kwargs["linewidths"] == 0.8

    # Bottom left subplot (2, 0): Predicted mask
    ax20_args, ax20_kwargs = mock_axes[2, 0].imshow.call_args
    np.testing.assert_array_equal(ax20_args[0], predicted_mask)
    assert ax20_kwargs["aspect"] == "auto"
    assert ax20_kwargs["interpolation"] == "None"
    assert ax20_kwargs["cmap"] == "Greens"
    mock_axes[2, 0].set_title.assert_called_once_with("Predicted mask", fontsize=12)

    # Bottom right subplot (2, 1): Predicted mask overlaid on NDVI image
    mock_axes[2, 1].imshow.assert_called_once_with(
        mock_ndvi, aspect="auto", interpolation="None", cmap=mock_ndvi_cmap
    )
    ax21_args, ax21_kwargs = mock_axes[2, 1].contour.call_args
    np.testing.assert_array_equal(ax21_args[0], predicted_mask)
    assert ax21_kwargs["levels"] == [0.5]
    assert ax21_kwargs["colors"] == "purple"
    assert ax21_kwargs["linewidths"] == 0.8
    mock_axes[2, 1].set_title.assert_called_once_with(
        "Predicted mask overlaid on NDVI image", fontsize=12
    )

    # Assertions for removing axes ticks but keep labels
    for mock_ax in mock_axes.flat:
        mock_ax.set_xticks.assert_called_once_with([])
        mock_ax.set_yticks.assert_called_once_with([])
    # Assertion for adding colorbar
    mock_fig.subplots_adjust.assert_called_once_with(
        top=0.96, bottom=0.05, left=0.01, right=0.89, wspace=0.1, hspace=0.1
    )
    # Assertions for colorbar setup
    mock_fig.add_axes.assert_called_once_with((0.90, 0.10, 0.02, 0.80))
    mock_fig.colorbar.assert_called_once_with(mock_ndvi_im, cax=mock_cbar_ax)
    mock_cbar.set_ticks.assert_called_once_with(np.arange(-1, 1.1, 0.2).tolist())
    mock_cbar.set_label.assert_called_once_with("NDVI", fontsize=12)
    # Assert for results annotation.
    mock_fig.text.assert_called_once_with(
        0.5, 0.02, results_annotation, ha="center", va="center", fontsize=12
    )
    # Assert for saving and closing figure.
    mock_plt_savefig.assert_called_once_with(basename + "_results.png")
    mock_plt_close.assert_called_once()


def test_plot_precision_recall_curve_unequal_input_lengths() -> None:
    """
    Check for correct raise of ValueError upon execution with unequal input array lengths.
    """
    with pytest.raises(
        ValueError,
        match="Precision, recall, and threshold arrays must have the same length.",
    ):
        # Create dummy inputs.
        precision_vals = np.array([0.8, 0.7, 0.6])
        recall_vals = np.array([0.6, 0.7, 0.8])
        threshold_vals = np.array([0.2, 0.5, 0.6, 0.8])

        # Call the test function.
        plot_precision_recall_curve(precision_vals, recall_vals, threshold_vals)


@patch("matplotlib.pyplot.close")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
@patch("amazon_seg_project.ops.plotting_utils.LineCollection")
@patch("matplotlib.pyplot.Normalize")
def test_plot_precision_recall_curve_execution(
    mock_plt_normalize: MagicMock,
    mock_LineCollection: MagicMock,
    mock_plt_subplots: MagicMock,
    mock_plt_tight_layout: MagicMock,
    mock_plt_savefig: MagicMock,
    mock_plt_close: MagicMock,
) -> None:
    """
    Test for correct plotting using mocked dependencies.
    """
    # Create dummy inputs.
    precision_vals = np.array([0.8, 0.7, 0.6])
    recall_vals = np.array([0.6, 0.7, 0.8])
    threshold_vals = np.array([0.2, 0.5, 0.8])
    max_f1_idx = 1
    n_positive_samples = 10
    n_samples = 100
    basename = "test_plot"

    # Expected intermediate compute products
    points = np.array([recall_vals, precision_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    baseline_precision = n_positive_samples / n_samples
    pr_auc = abs(np.trapz(precision_vals, recall_vals))

    # Set up mock dependencies.
    mock_fig = MagicMock(name="mock-figure")
    mock_ax = MagicMock(name="mock-ax")
    mock_plt_subplots.return_value = (mock_fig, mock_ax)

    mock_lc = MagicMock(name="mock-lc")
    mock_LineCollection.return_value = mock_lc

    mock_line = MagicMock(name="mock-colored-line")
    mock_ax.add_collection.return_value = mock_line

    # Call the test function.
    plot_precision_recall_curve(
        precision_vals,
        recall_vals,
        threshold_vals,
        n_positive_samples=n_positive_samples,
        n_samples=n_samples,
        basename=basename,
    )

    # Assertions
    mock_plt_normalize.assert_called_once_with(vmin=0.0, vmax=1.0, clip=True)
    # LineCollection initialization
    args, kwargs = mock_LineCollection.call_args
    np.testing.assert_array_equal(args[0], segments)
    assert kwargs["cmap"] == "cividis"
    assert kwargs["norm"] == mock_plt_normalize.return_value
    mock_lc.set_array.assert_called_once_with(threshold_vals)
    mock_lc.set_linewidth.assert_called_once_with(2)
    # Subplot creation
    mock_plt_subplots.assert_called_once_with(nrows=1, ncols=1, figsize=(6, 5))
    # Dotted line for baseline always positive model
    mock_ax.hlines.assert_called_once_with(
        y=baseline_precision, xmin=0, xmax=1, linestyle=":", color="k"
    )
    mock_ax.annotate.assert_any_call(
        "Baseline always positive model",
        xy=(0.2, baseline_precision + 0.01),
        xycoords="data",
        fontsize=12,
    )
    # Assertion for colored line creation in ax.
    mock_ax.add_collection.assert_called_once_with(mock_lc)
    mock_fig.colorbar.assert_called_once_with(mock_line, ax=mock_ax, label="Threshold")
    # Annotation for AUC
    mock_ax.annotate.assert_any_call(
        f"AUC = {pr_auc:.3f}",
        xy=(0.3, 0.8),
        xycoords="data",
        fontsize=12,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )
    # Assertion for plotting marker at data point with the highest F1 score
    mock_ax.plot.assert_called_once_with(
        recall_vals[max_f1_idx],
        precision_vals[max_f1_idx],
        marker="*",
        markeredgecolor="black",
        markerfacecolor="darkorange",
        markersize=12,
        linestyle="None",
    )
    # Assertion for axes limits
    mock_ax.set_xlim.assert_called_once_with(0, 1)
    mock_ax.set_ylim.assert_called_once_with(0, 1.05)
    # Assertion for axes labels
    mock_ax.set_xlabel.assert_called_once_with("Recall", fontsize=12)
    mock_ax.set_ylabel.assert_called_once_with("Precision", fontsize=12)
    # Check for a single call of plt.tight_layout().
    mock_plt_tight_layout.assert_called_once()
    # Assert for saving and closing figure.
    mock_plt_savefig.assert_called_once_with(basename + "_precision_recall_curve.png")
    mock_plt_close.assert_called_once()
