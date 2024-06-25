from src.plantseg_fractal_tasks.import_from_plantseg_h5 import import_from_plantseg_h5
from src.plantseg_fractal_tasks.plantseg_workflow import plantseg_workflow

zarr_url = "/Users/locerr/data/"
h5_path = "/Users/locerr/data/sample_ovules.h5"
print("Running import_from_plantseg_h5")
import_from_plantseg_h5(zarr_urls=[""], zarr_dir=zarr_url, h5_path=h5_path, label_key="label", levels=3)
plantseg_workflow(zarr_url="/Users/locerr/data/sample_ovules.zarr/raw/", unet_name="plantseg_ovules")
