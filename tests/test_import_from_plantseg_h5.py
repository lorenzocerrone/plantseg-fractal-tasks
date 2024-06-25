import shutil
from pathlib import Path

import pytest
from devtools import debug
import h5py
import numpy as np

from fractal_tasks_core.channels import ChannelInputModel
from plantseg_fractal_tasks.import_from_plantseg_h5 import import_from_plantseg_h5


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> dict[str, Path]:
    """
    Create a test-data folder into a temporary folder.
    """
    source_dir = tmp_path / "samples_h5"
    source_dir.mkdir()
    file = source_dir / "sample.h5"
    zarr_dir = tmp_path / "zarr"
    # Create a test-data folder
    with h5py.File(file, "w") as f:
        f.create_dataset("raw", data=np.random.rand(10, 10, 10))
        f["raw"].attrs["element_size_um"] = [1.0, 1.0, 1.0]
        f.create_dataset("label", data=np.random.randint(0, 2, (10, 10, 10)))

    return {"file": file, "zarr_dir": zarr_dir, "source_dir": source_dir}


def test_import_from_plantseg_h5(test_data_dir):
    zarr_dir = test_data_dir["zarr_dir"]
    h5_path = test_data_dir["file"]

    updates_list = import_from_plantseg_h5(
        zarr_urls=[""], zarr_dir=str(zarr_dir), h5_path=str(h5_path), label_key="label", levels=3
    )
    print(updates_list)
    assert zarr_dir.exists()
