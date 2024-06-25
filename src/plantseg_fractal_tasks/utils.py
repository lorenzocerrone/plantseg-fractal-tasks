import fractal_tasks_core
from pydantic.v1 import validator, BaseModel
from fractal_tasks_core.ngff.specs import Axis, Dataset, ScaleCoordinateTransformation
from fractal_tasks_core.ngff.specs import Multiscale, NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
import numpy as np
import zarr
from pathlib import Path
from typing import Optional, Callable

VALID_AXIS_NAMES = ["t", "c", "z", "x", "y"]
VALID_TIME_UNITS = {
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
}
VALID_SPACE_UNITS = {
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
}
AXIS_TYPES = {"t": "time", "c": "channel", "z": "space", "x": "space", "y": "space"}


class AxisModel(BaseModel):
    """
    AxisModel is a Pydantic model for the voxel size of an image.

    Args:
        axis_name (str): Name of the axis.
        element_size (float): Size of the element.
        unit (str): Unit of the element size.
        scaling_factor (float): Scaling factor used to generate the image pyramids.
            Default is 1.0. If the axis represents a channel, the scaling factor is 1.0.
    """

    axis_name: str
    element_size: float
    unit: str
    scaling_factor: float = 1.0

    @validator("axis_name")
    def validate_axis_name(cls, v):
        assert v in VALID_AXIS_NAMES
        return v

    @validator("element_size")
    def validate_element_size(cls, v, **kwargs):
        assert v > 0, "Element size must be a positive number."
        return v

    @validator("unit")
    def validate_element_unit(cls, v, values):
        axis_name = values.get("axis_name")
        if AXIS_TYPES[axis_name] == "time":
            assert v in VALID_TIME_UNITS
        if AXIS_TYPES[axis_name] == "space":
            assert v in VALID_SPACE_UNITS
        return v

    @validator("scaling_factor")
    def validate_scaling_factor(cls, v, values):
        assert v > 0, "Scaling factor must be a positive number."
        axis_name = values.get("axis_name")
        if AXIS_TYPES[axis_name] == "channel":
            assert v == 1.0
        return v

    def element_size_at_level(self, level: int = 0) -> float:
        assert level >= 0, "Level must be a non-negative integer."
        assert isinstance(level, int), "Level must be an integer."

        if level == 0:
            return self.element_size
        return self.element_size_at_level(level - 1) * self.scaling_factor


def axis_models_from_element_sizes(image_ndim: int, element_size: np.ndarray) -> list[AxisModel]:

    if image_ndim == 5:
        assert element_size.shape[0] == 3, "Invalid number of elements in the element size array."
        new_element_size = [1.0, 1, 0, *element_size]
        axis_names = ["t", "c", "z", "x", "y"]
        voxel_size_unit = ["seconds", "", "micrometer", "micrometer", "micrometer", "micrometer"]
        scaling = [1.0, 1.0, 1.0, 2.0, 2.0]

    elif image_ndim == 4:
        assert element_size.shape[0] == 3, "Invalid number of elements in the element size array."
        new_element_size = [1.0, *element_size]
        axis_names = ["c", "z", "x", "y"]
        voxel_size_unit = ["", "micrometer", "micrometer", "micrometer"]
        scaling = [1.0, 1.0, 2.0, 2.0]

    elif image_ndim == 3:
        assert element_size.shape[0] == 3, "Invalid number of elements in the element size array."
        new_element_size = element_size.tolist()
        axis_names = ["z", "x", "y"]
        voxel_size_unit = ["micrometer", "micrometer", "micrometer"]
        scaling = [1.0, 2.0, 2.0]

    elif image_ndim == 2:
        assert element_size.shape[0] == 2, "Invalid number of elements in the element size array."
        new_element_size = element_size.tolist()
        axis_names = ["x", "y"]
        voxel_size_unit = ["micrometer", "micrometer"]
        scaling = [2.0, 2.0]

    else:
        raise ValueError(
            "Invalid number of dimensions in the raw image. Expected 3 or 4. (timelapses are not supported)"
        )

    return [
        AxisModel(axis_name=n, element_size=es, unit=u, scaling_factor=sc)
        for n, es, u, sc in zip(axis_names, new_element_size, voxel_size_unit, scaling)
    ]


def build_multiscale_metadata(axis_models: list[AxisModel], num_levels: int = 1, name: str = "TBD") -> Multiscale:

    axis_names = [model.axis_name for model in axis_models]
    voxel_size_unit = [model.unit for model in axis_models]

    axes_metadata = [Axis(name=n, type=AXIS_TYPES[n], unit=u) for n, u in zip(axis_names, voxel_size_unit)]

    datasets = []
    for i in range(num_levels):
        elment_sizes = [model.element_size_at_level(i) for model in axis_models]
        _dataset = Dataset(
            path=str(i), coordinateTransformations=[ScaleCoordinateTransformation(type="scale", scale=elment_sizes)]
        )
        datasets.append(_dataset)

    return Multiscale(version=fractal_tasks_core.__OME_NGFF_VERSION__, name=name, axes=axes_metadata, datasets=datasets)


def save_multiscale_image(
    zarr_url: Path,
    image: np.ndarray,
    metadata: NgffImageMeta,
    aggregation_function: Optional[Callable] = None,
    mode="a",
):
    ome_zarr = zarr.open(str(zarr_url), mode=mode)
    assert isinstance(ome_zarr, zarr.Group), "Zarr image must be a group."
    ome_zarr.create_dataset("0", data=image, dimension_separator="/")
    ome_zarr.attrs.update(metadata.dict())
    chunksize = ome_zarr.get("0")
    if chunksize is not None:
        chunksize = chunksize.chunks
    else:
        chunksize = None

    if aggregation_function is None:
        aggregation_function = np.mean

    build_pyramid(
        zarrurl=zarr_url,
        overwrite=True,
        chunksize=chunksize,
        num_levels=metadata.num_levels,
        coarsening_xy=metadata.coarsening_xy,
        aggregation_function=aggregation_function,
    )
