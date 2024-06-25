"""
This is the Python module for my_task
"""

from typing import Any
import zarr
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from pydantic.v1.decorator import validate_arguments
from pathlib import Path
from plantseg_fractal_tasks.utils import save_multiscale_image
import numpy as np
from plantseg.predictions.functional import unet_predictions
from plantseg.segmentation.functional import gasp, dt_watershed, mutex_ws, multicut
from enum import Enum
from typing import Callable
from functools import partial


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class SegmentationType(str, Enum):
    gasp = "gasp"
    mutex_ws = "mutex_ws"
    multicut = "multicut"

    def load_segmentation(self, **kwargs) -> Callable:
        if self == SegmentationType.gasp:
            return partial(gasp, **kwargs)
        elif self == SegmentationType.mutex_ws:
            return partial(mutex_ws, **kwargs)
        elif self == SegmentationType.multicut:
            return partial(multicut, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation type: {self}")


@validate_arguments
def plantseg_workflow(
    *,
    zarr_url: str,
    unet_name: str = "generic_confocal",
    beta: float = 0.6,
    post_minsize: int = 100,
    device: Device = Device.cuda,
    ws_threshold: float = 0.5,
    segmentation_type: SegmentationType = SegmentationType.gasp,
    patch: tuple[int, int, int] = (80, 160, 160),
) -> dict[str, list[dict[str, Any]]]:
    """
    This function imports a raw image and a label image from a PlantSeg HDF5 file
    to a OME-Zarr image.

    Args:
        zarr_url: The URL of the Zarr file.
        unet_name: The name of the U-Net model to use.
        beta: The beta parameter for the GASP algorithm.
        post_minsize: The minimum size of the post-processed segments.
        device: The device to use for the U-Net model.
        ws_threshold: The threshold for the watershed algorithm.
        segmentation_type: The type of segmentation to use.
        patch: The patch size to use for the U-Net model.
    """

    raw_image = zarr.open(zarr_url + "/0")
    metadata = load_NgffImageMeta(zarr_url)
    assert isinstance(raw_image, zarr.Array), "Raw image must be a zarr array."

    raw_image = raw_image[...]
    predictions = unet_predictions(
        raw_image,
        model_name=unet_name,
        model_id=None,
        patch=patch,
        single_batch_mode=True,
        device=device.value,
        model_update=False,
        disable_tqdm=True,
        handle_multichannel=False,
    )

    superpixels = dt_watershed(predictions, threshold=ws_threshold)
    segmentation = gasp(
        boundary_pmaps=predictions, superpixels=superpixels, beta=beta, post_minsize=post_minsize, n_threads=1
    ).astype("uint16")

    new_zarr_name = Path(zarr_url).parent / f"predictions"
    save_multiscale_image(new_zarr_name, predictions, metadata, mode="w")

    label_group_name = Path(zarr_url).parent / f"predictions/labels/"
    label_group = zarr.open_group(label_group_name, mode="a")

    if "labels" in label_group.attrs:
        assert isinstance(label_group.attrs["labels"], list), "Labels must be a list."
        label_group.attrs["labels"].append("gasp")
    else:
        label_group.attrs["labels"] = ["gasp"]

    new_label_name = Path(zarr_url).parent / f"predictions/labels/gasp"
    save_multiscale_image(new_label_name, segmentation, metadata, mode="a", aggregation_function=np.max)
    return {
        "image_list_updates": [
            {
                "zarr_url": str(new_zarr_name),
                "origin": zarr_url,
                "types": {"is_3D": True, "is_prediction": True},
            }
        ]
    }


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=plantseg_workflow)
