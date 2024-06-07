from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    encode_segmentation,
)
from pathlib import Path

ZARR_FILEPATH = (
    r"/media/starfish/LargeSSD/data/cryoET/data/00004_actin_ground_truth_zarr"
)

OUTPUT_DIR = Path(r"/media/starfish/LargeSSD/data/cryoET/data/actin_converted_new/")

encode_segmentation(
    ZARR_FILEPATH, OUTPUT_DIR, resolution=(1.0, 1.0, 1.0), include_mesh=True
)
