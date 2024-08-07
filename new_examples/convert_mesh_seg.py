from pathlib import Path
import logging
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    encode_segmentation,
)

# Set up logging to debug level
logging.basicConfig(level=logging.INFO, force=True)

INPUT_FILENAME = r"/media/starfish/LargeSSD/data/cryoET/data/00004_MT_ground_truth_zarr"
OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/new_MT_converted_mesh/")


encode_segmentation(
    filename=INPUT_FILENAME,
    output_path=OUTPUT_PATH,
    resolution=(1.048, 1.048, 1.048),
    max_lod=2,
    include_mesh=True,
    delete_existing=True,
)
