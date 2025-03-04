import json
from pathlib import Path
import logging
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    encode_segmentation,
)
from cryoet_data_portal_neuroglancer.state_generator import (
    generate_segmentation_mask_layer,
)

# Set up logging to debug level
logging.basicConfig(level=logging.INFO, force=True)

INPUT_FILENAME = r"/media/starfish/LargeSSD/data/cryoET/data/00004_MT_ground_truth_zarr"
OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/new_MT_converted_mesh/")
SOURCE = "http://localhost:1337"

encode_segmentation(
    filename=INPUT_FILENAME,
    output_path=OUTPUT_PATH,
    resolution=(1.048, 1.048, 1.048),
    max_lod=2,
    include_mesh=True,
    delete_existing=True,
    fast_bounding_box=True,
    max_simplification_error_in_voxels=4,
    min_mesh_chunk_dim=8,
)

layer_json = generate_segmentation_mask_layer(
    source=SOURCE,
    name="Test segmentation with mesh",
    scale=1.048,
    color="#FF0000",
)
json.dump(layer_json, open("segmentation_with_mesh.json", "w"), indent=2)

# To serve the segmentation, can run something like:
from cloudvolume import CloudVolume

volume = CloudVolume(f"file://{OUTPUT_PATH.resolve()}")
volume.viewer(port=1337)
