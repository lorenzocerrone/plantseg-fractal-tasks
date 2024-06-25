"""
This is the Python module for my_task
"""

import logging
from typing import Any
from typing import Optional

import numpy as np
import glob
import zarr
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from pydantic.v1.decorator import validate_arguments
import h5py
from pathlib import Path
from fractal_tasks_core.ngff.specs import NgffImageMeta
from plantseg_fractal_tasks.utils import (
    AxisModel,
    axis_models_from_element_sizes,
    build_multiscale_metadata,
    save_multiscale_image,
)


def h5_to_omezarr(
    h5_path: Path,
    zarr_dir: Path,
    raw_key: str = "raw",
    label_key: Optional[str] = None,
    voxel_size_model: Optional[list[AxisModel]] = None,
    levels: int = 3,
):
    logging.info(f"Processing {h5_path=}")

    with h5py.File(h5_path, "r") as f:

        raw = f.get(raw_key, None)
        if raw is None:
            raise ValueError(f"Key {raw_key} not found in the HDF5 file.")

        assert isinstance(raw, h5py.Dataset), "Raw image must be a dataset."
        element_size = raw.attrs.get("element_size_um", None)
        raw = raw[...]

        labels = f.get(label_key, None)
        if labels is not None:
            assert isinstance(labels, h5py.Dataset), "Label image must be a dataset."
            labels = labels[...]

    if voxel_size_model is None and element_size is not None:
        voxel_size_model = axis_models_from_element_sizes(raw.ndim, element_size)
    else:
        raise ValueError("Voxel size model is required if element size is not present in the HDF5 file")

    multiscale_metadata = build_multiscale_metadata(voxel_size_model, levels, name="OME-Zarr")
    metadata = NgffImageMeta(multiscales=[multiscale_metadata])

    new_zarr_name = zarr_dir / f"{Path(h5_path).stem}.zarr/raw"

    save_multiscale_image(
        zarr_url=new_zarr_name,
        image=raw,
        metadata=metadata,
        aggregation_function=np.mean,
        mode="w",
    )

    if labels is not None:
        label_group_name = Path(new_zarr_name).parent / f"raw/labels/"
        label_group = zarr.open_group(label_group_name, mode="a")

        if "labels" in label_group.attrs:
            assert isinstance(label_group.attrs["labels"], list), "Labels must be a list."
            label_group.attrs["labels"].append("grount_truth")
        else:
            label_group.attrs["labels"] = ["ground_truth"]

        new_zarr_name = zarr_dir / f"{Path(h5_path).stem}.zarr/raw/labels/ground_truth"
        save_multiscale_image(
            zarr_url=new_zarr_name,
            image=labels,
            metadata=metadata,
            aggregation_function=np.max,
            mode="a",
        )

    return dict(
        zarr_url=str(new_zarr_name),
        attributes=dict(),
        types=dict(is_3D=True if raw.ndim >= 3 else False),
    )


def parse_h5_path(h5_path: str) -> list[Path]:
    h5_pathlib_path = Path(h5_path)

    if h5_pathlib_path.is_file():
        assert h5_pathlib_path.suffix == ".h5", "File must have the .h5 extension."
        return [h5_pathlib_path]

    elif h5_pathlib_path.is_dir():
        h5_to_convert = list(h5_pathlib_path.glob("*.h5"))
        if not h5_to_convert:
            raise ValueError(f"No HDF5 files found in {h5_pathlib_path}.")
        return h5_to_convert

    elif not h5_pathlib_path.exists():
        h5_to_convert = glob.glob(h5_path)
        h5_to_convert = [Path(path) for path in h5_to_convert]
        if not h5_to_convert:
            raise ValueError(f"{h5_path} does not exist, and no files match the glob pattern.")
        return h5_to_convert

    raise ValueError(f"Invalid path {h5_path}. Must be a file, a directory, or a valid glob pattern.")


@validate_arguments
def import_from_plantseg_h5(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    h5_path: str,
    raw_key: str = "raw",
    label_key: Optional[str] = None,
    voxel_size_model: Optional[list[AxisModel]] = None,
    levels: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """
    This function imports a raw image and a label image from a PlantSeg HDF5 file
    to a OME-Zarr image.

    Args:
        zarr_urls: List of URLs to the OME-Zarr image.
        zarr_dir: Absolute path to the OME-Zarr image.
        h5_path: Can be the absolute path to the PlantSeg HDF5 file, a directory containing a bunch of h5 files, or a valid glob pattern.
        raw_key: Key of the raw image in the HDF5 file.
        label_key: Key of the label image in the HDF5 file.
        voxel_size_model: List of VoxelsizeModel objects, containing the element size, axis name and unit.
        levels: Number of levels in the multiscale image.
    """
    zarr_dir_pathlib_path = Path(zarr_dir)
    h5_to_convert = parse_h5_path(h5_path)

    image_list_updates = [
        h5_to_omezarr(path, zarr_dir_pathlib_path, raw_key, label_key, voxel_size_model, levels)
        for path in h5_to_convert
    ]

    return dict(image_list_updates=image_list_updates)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=import_from_plantseg_h5)
