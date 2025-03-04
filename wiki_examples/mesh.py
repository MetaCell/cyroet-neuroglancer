import json
from pathlib import Path

from cryoet_data_portal_neuroglancer.io import load_glb_file
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_multiresolution_mesh,
)
from cryoet_data_portal_neuroglancer.state_generator import (
    generate_oriented_point_mesh_layer,
)

MESH_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/meshes-oriented/atpase.glb"
)
OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")
SOURCE = "http://localhost:1337"

# Load and convert the mesh to a precomputed format
scene = load_glb_file(MESH_PATH)
generate_multiresolution_mesh(
    scene, OUTPUT_PATH / "glb_mesh_converted", min_mesh_chunk_dim=16, max_lod=2
)

# Create the JSON layer state for the mesh
layer_json = generate_oriented_point_mesh_layer(
    source=SOURCE,
    name="Test mesh",
    color=None,
    scale=0.784 * 1e-9,
    mesh_render_scale=20.0,
    visible_segments=(2, 4, 1, 3),
)
json.dump(layer_json, open("mesh.json", "w"), indent=2)

# To serve the mesh, can run something like:
from cloudvolume import CloudVolume

volume = CloudVolume(f"file://{(OUTPUT_PATH / 'glb_mesh_converted').resolve()}")
volume.viewer(port=1337)
