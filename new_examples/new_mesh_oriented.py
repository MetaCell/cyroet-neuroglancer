import json

from cryoet_data_portal_neuroglancer.precompute.instance_mesh import (
    encode_oriented_mesh,
)
from cryoet_data_portal_neuroglancer.state_generator import (
    generate_oriented_point_mesh_layer,
)
from cryoet_data_portal_neuroglancer.io import load_glb_file, load_oriented_point_data
from pathlib import Path
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_mesh_from_lods,
)

# Optional: set up logging to debug
# import logging
# logging.basicConfig(level=logging.DEBUG, force=True)

# Setup input and output paths
ANNOTATION_JSON_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/Annotations-test-run/100-proton_transporting_atp_synthase_complex-1.0.json"
)
GLB_MESH_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/meshes-oriented/atpase.glb"
)
OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")
SOURCE = "http://localhost:1337"


# Load the oriented point data and the mesh
metadata, data = load_oriented_point_data(
    ANNOTATION_JSON_PATH,
    ANNOTATION_JSON_PATH.with_name(ANNOTATION_JSON_PATH.stem + "_orientedpoint.ndjson"),
)
scene = load_glb_file(GLB_MESH_PATH)

# Convert the mesh to a precomputed format oriented mesh
oriented_mesh_at_each_lod = encode_oriented_mesh(
    scene,
    data,
    max_lod=2,
    max_faces_for_first_lod=10e6,
    decimation_aggressiveness=5.5,
)
generate_mesh_from_lods(
    oriented_mesh_at_each_lod, OUTPUT_PATH / "meshoutput", min_mesh_chunk_dim=2
)

# Create and save the related JSON state for that layer
layer_json = generate_oriented_point_mesh_layer(
    source=SOURCE,
    name="Test oriented point mesh",
    color="#FF0000",
    scale=0.784 * 1e-9,
)
json.dump(layer_json, open("oriented_point_mesh_layer.json", "w"), indent=2)

# To serve and view the files, you can run something like:
from cloudvolume import CloudVolume
from cryoet_data_portal_neuroglancer.state_generator import combine_json_layers

full_json_state = combine_json_layers([layer_json], 0.784 * 1e-9)
json.dump(full_json_state, open("full_json_state.json", "w"), indent=2)
# Paste above state into the viewer

volume = CloudVolume(f"file://{(OUTPUT_PATH / 'meshoutput').resolve()}")
volume.viewer(port=1337)
