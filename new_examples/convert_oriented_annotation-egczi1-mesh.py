import json
import logging

from cryoet_data_portal_neuroglancer.precompute.points import encode_annotation
from cryoet_data_portal_neuroglancer.precompute.instance_mesh import (
    encode_oriented_mesh,
)
from cryoet_data_portal_neuroglancer.state_generator import (
    generate_oriented_point_layer,
    combine_json_layers,
    generate_segmentation_mask_layer,
)
from cryoet_data_portal_neuroglancer.io import load_glb_file, load_oriented_point_data
from pathlib import Path
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_sharded_mesh_from_lods,
)


JSON_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/Annotations-test-run/100-proton_transporting_atp_synthase_complex-1.0.json"
)

MESH_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/meshes-oriented/atpase.glb"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")

metadata, data = load_oriented_point_data(
    JSON_PATH, JSON_PATH.with_name(JSON_PATH.stem + "_orientedpoint.ndjson")
)

encode_annotation(
    data, metadata, OUTPUT_PATH, 0.784 * 1e-9, shard_by_id=(0, 10), is_oriented=True
)
logging.basicConfig(level=logging.INFO, force=True)
scene = load_glb_file(MESH_PATH)

copy_pasted_lods = encode_oriented_mesh(
    scene,
    data,
    num_lods=3,
    max_faces_for_first_lod=10e10,
)
generate_sharded_mesh_from_lods(copy_pasted_lods, OUTPUT_PATH / "meshoutput")

SOURCE = "http://127.0.0.1:9000/converted-01122021/"

output = generate_oriented_point_layer(
    source=SOURCE,
    name="Test Oriented Points",
    color="#FFFFFF",
    point_size_multiplier=0.5,
    line_width=2.0,
    is_visible=True,
    is_instance_segmentation=False,
    scale=0.784 * 1e-9,
)

output2 = generate_segmentation_mask_layer(
    source="http://localhost:1337",
    name="Test Mesh",
    color="#FF0000",
    scale=0.784 * 1e-9,
)
layer_json = combine_json_layers([output, output2], 0.784, units="nm")
json.dump(layer_json, open("oriented_point_layer.json", "w"), indent=2)
