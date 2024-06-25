"""
This is the Python module for my_task
"""

from typing import Any
import zarr

from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from pydantic.v1.decorator import validate_arguments
from pathlib import Path
from plantseg_fractal_tasks.utils import save_multiscale_image
import numpy as np
from plantseg.dataprocessing.functional import image_gaussian_smoothing, image_rescale, image_crop
from plantseg.predictions.functional import unet_predictions
from plantseg.segmentation.functional import gasp, dt_watershed, mutex_ws, multicut
from functools import partial
from plantseg_fractal_tasks.input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
    PlantSegPreprocessingModel,
)


def load_zarr(zarr_url: str, channel, level: int = 0) -> tuple[np.ndarray, NgffImageMeta]:
    raw_image = zarr.open(zarr_url + f"/{level}")
    metadata = load_NgffImageMeta(zarr_url)

    # This a workaround before the new OME-Zarr io implementation
    axis = metadata.multiscales[0].axes
    slices = []
    for i, ax in enumerate(axis):
        if ax.type == "c" or ax.type == "channel":
            _slice = channel
        else:
            _slice = slice(None)
        slices.append(_slice)

    assert isinstance(raw_image, zarr.Array), "Raw image must be a zarr array."
    raw_image = raw_image[tuple(slices)]
    return raw_image, metadata


@validate_arguments
def plantseg_workflow(
    *,
    zarr_url: str,
    channel: int,
    preprocessing_model: PlantSegPreprocessingModel,
    prediction_model: PlantSegPredictionsModel,
    segmentation_model: PlantSegSegmentationModel,
) -> dict[str, list[dict[str, Any]]]:
    """
    This function imports a raw image and a label image from a PlantSeg HDF5 file
    to a OME-Zarr image.

    Args:
        zarr_url: The URL of the Zarr file.
        prediction: The prediction model to use.
        ws_threshold: The threshold for the watershed.
        segmentation_type: The segmentation type to use.
        beta: The beta value.
        post_minsize: The minimum size.
    """

    raw_image, metadata = load_zarr(zarr_url, channel)
    image_list_updates = []

    if preprocessing_model.skip is False:
        if preprocessing_model.rescaling_factor is not None:
            raw_image = image_rescale(raw_image, preprocessing_model.rescaling_factor, order=1)

        if preprocessing_model.sigma_gaussian_filter is not None:
            raw_image = image_gaussian_smoothing(raw_image, preprocessing_model.sigma_gaussian_filter)

        if preprocessing_model.manual_cropping is not None:
            raw_image = image_crop(raw_image, preprocessing_model.manual_cropping)

    if prediction_model.skip is False:

        if prediction_model.model_source == "PlantSegZoo":
            model_name = prediction_model.plantsegzoo_name
            model_id = None
        else:
            model_name = None
            model_id = prediction_model.bioimageio_name

            predictions = unet_predictions(
                raw_image,
                model_name=model_name,
                model_id=model_id,
                patch=prediction_model.patch,
                single_batch_mode=True,
                device=prediction_model.device,
                model_update=False,
                disable_tqdm=True,
                handle_multichannel=False,
            )

            if prediction_model.save_results:
                # Save the predictions
                new_zarr_name = Path(zarr_url).parent / f"predictions"
                save_multiscale_image(new_zarr_name, predictions, metadata, mode="w")

                image_list_updates.append(
                    {
                        "zarr_url": str(new_zarr_name),
                        "origin": zarr_url,
                        "types": {"is_3D": True, "is_prediction": True},
                    }
                )

    else:
        predictions = raw_image

    if segmentation_model.skip is False:
        # run distance transform watershed
        segmentation = dt_watershed(predictions, threshold=segmentation_model.ws_threshold)

        if segmentation_model.segmentation_type == "gasp":
            segmentation_func = partial(gasp, n_threads=1)

        elif segmentation_model.segmentation_type == "mutex_ws":
            segmentation_func = partial(mutex_ws, n_threads=1)

        elif segmentation_model.segmentation_type == "multicut":
            segmentation_func = multicut

        elif segmentation_model.segmentation_type == "dt_watershed":
            # avoid re-running dt_watershed
            segmentation_func = lambda *args, **kwargs: segmentation
        else:
            raise ValueError("Invalid segmentation type.")

        segmentation = segmentation_func(
            boundary_pmaps=predictions,
            superpixels=segmentation,
            beta=segmentation_model.beta,
            post_minsize=segmentation_model.post_minsize,
        )

        label_group_name = Path(zarr_url).parent / f"predictions/labels/"
        label_group = zarr.open_group(label_group_name, mode="a")

        if "labels" in label_group.attrs:
            assert isinstance(label_group.attrs["labels"], list), "Labels must be a list."
            label_group.attrs["labels"].append(f"plantseg_{segmentation_model.segmentation_type}")
        else:
            label_group.attrs["labels"] = [f"plantseg_{segmentation_model.segmentation_type}"]

        new_label_name = Path(zarr_url).parent / f"predictions/labels/plantseg_{segmentation_model.segmentation_type}"
        save_multiscale_image(new_label_name, segmentation, metadata, mode="a", aggregation_function=np.max)

        image_list_updates.append(
            {
                "zarr_url": str(new_label_name),
                "origin": zarr_url,
                "types": {"is_3D": True},
            }
        )

    return dict(image_list_updates=image_list_updates)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=plantseg_workflow)
