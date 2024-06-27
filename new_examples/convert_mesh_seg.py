from pathlib import Path
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    encode_segmentation,
)

INPUT_FILENAME = (
    r"/media/starfish/LargeSSD/data/cryoET/data/00004_actin_ground_truth_zarr"
)
OUTPUT_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/new_actin_converted_mesh_no_res/"
)


encode_segmentation(
    filename=INPUT_FILENAME,
    output_path=OUTPUT_PATH,
    resolution=(1.048, 1.048, 1.048),
    num_lod=3,
    include_mesh=True,
    delete_existing=True,
)
