import logging
from pathlib import Path

import numpy as np
import trimesh
from cryoet_data_portal_neuroglancer.io import load_glb_file
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_standalone_sharded_multiresolution_mesh,
    generate_sharded_mesh_from_lods,
    decimate_mesh,
)
from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh

JSON_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/Annotations-test-run/100-proton_transporting_atp_synthase_complex-1.0.json"
)

MESH_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/meshes-oriented/atpase.glb"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")

logging.basicConfig(level=logging.INFO, force=True)
scene = load_glb_file(MESH_PATH)

## Simple conversion
# generate_standalone_sharded_multiresolution_mesh(
#     scene, OUTPUT_PATH / "raw_glb_mesh", 2, min_mesh_chunk_dim=4
# )

## Decimate first conversion
lods = []
for lod in decimate_mesh(scene.dump(concatenate=True), 2, as_trimesh=True):
    lod.apply_scale(0.1)
    scene = trimesh.Scene()
    translatation = [100, 200, 4.4]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotate_and_translate_mesh(lod, scene, 0, rotation, translatation)
    translatation = [10, 199, 5.4]
    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotate_and_translate_mesh(lod, scene, 1, rotation, translatation)
    lods.append(scene)
full_mesh = trimesh.Scene(lods[0].dump(concatenate=True))
generate_standalone_sharded_multiresolution_mesh(
    full_mesh, OUTPUT_PATH / "glb_shard", 1, min_mesh_chunk_dim=4
)

generate_sharded_mesh_from_lods(
    lods, OUTPUT_PATH / "raw_glb_mesh", min_mesh_chunk_dim=4
)
