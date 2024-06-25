from pydantic.v1 import BaseModel
from enum import StrEnum
from plantseg.models.zoo import model_zoo
from typing import Optional


class Device(StrEnum):
    cpu = "cpu"
    cuda = "cuda"


DynamicallyGeneratedModel = StrEnum(
    "DynamicallyGeneratedModel", {model_name: model_name for model_name in model_zoo.list_models()}
)


DynamicBioIOModels = StrEnum(
    "DynamicBioIOModels", {model_name: model_name for model_name in model_zoo.get_bioimageio_zoo_all_model_names()}
)

DynamicallyGeneratedModel.__doc__ = "Select a model from the PlantSeg Zoo."
DynamicBioIOModels.__doc__ = "Select a model from the BioImageIO Zoo."


class ModelsPool(StrEnum):
    """
    Select if the model is sourced from PlantSegZoo or BioImageIO.
    """

    PlantSegZoo = "PlantSegZoo"
    BioImageIO = "BioImageIO"


class PlantSegPredictionsModel(BaseModel):
    """
    Input model for PlantSeg predictions.

    Args:
        model_source (ModelsPool): The source of the model.
        plantsegzoo_name (DynamicallyGeneratedModel): The model name from the PlantSeg Zoo.
        bioimageio_name (DynamicBioIOModels): The model name from the BioImageIO Zoo.
        device (Device): The device to use for predictions.
        patch (tuple[int, int, int]): The patch size.
        save_results (bool): Whether to save the results.
    """

    model_source: ModelsPool = ModelsPool.PlantSegZoo
    plantsegzoo_name: DynamicallyGeneratedModel = model_zoo.list_models()[0]  # type: ignore
    bioimageio_name: DynamicBioIOModels = model_zoo.get_bioimageio_zoo_all_model_names()[0]  # type: ignore
    device: Device = Device.cuda
    patch: tuple[int, int, int] = (80, 160, 160)
    save_results: bool = False
    skip: bool = False


class SegmentationType(StrEnum):
    gasp = "gasp"
    mutex_ws = "mutex_ws"
    multicut = "multicut"
    dt_watershed = "dt_watershed"


class PlantSegSegmentationModel(BaseModel):
    """
    Input model for PlantSeg segmentations.

    Args:
        ws_threshold (float): The threshold for the watershed.
        segmentation_type (SegmentationType): The segmentation type to use.
        beta (float): The beta value.
        post_minsize (int): The minimum size.
    """

    ws_threshold: float = 0.5
    segmentation_type: SegmentationType = SegmentationType.gasp
    beta: float = 0.6
    post_minsize: int = 100
    skip: bool = False


class PlantSegPreprocessingModel(BaseModel):
    """
    Define the optional preprocessing steps to apply to the raw image.

    Args:
        rescaling_factor (tuple[float, float, float]): The rescaling factor for the raw image.
            For example (0.5, 0.5, 1.0) will rescale the x and y axes by 0.5 and keep the z axis as is.
        sigma_gaussian_filter (float): The sigma value for the Gaussian filter. If None, no filter is applied.
        manual_cropping (str): The manual cropping value. Define a manual slice in a numpy-like format.
            For example, "[0:100, 0:100, 0:100]" will crop the image from 0 to 100 in all axes.
        skip (bool): Whether to skip the preprocessing steps.
    """

    rescaling_factor: Optional[tuple[float, float, float]] = None
    sigma_gaussian_filter: Optional[float] = None
    manual_cropping: Optional[str] = None

    skip: bool = True
