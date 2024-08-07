import logging
from pathlib import Path

import trimesh
from cryoet_data_portal_neuroglancer.precompute.instance_mesh import (
    scale_and_decimate_mesh,
)
from cryoet_data_portal_neuroglancer.io import load_glb_file
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_standalone_sharded_multiresolution_mesh,
)


JSON_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/Annotations-test-run/100-proton_transporting_atp_synthase_complex-1.0.json"
)

MESH_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/meshes-oriented/atpase.glb"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")

logging.basicConfig(level=logging.INFO, force=True)
scene = load_glb_file(MESH_PATH)
scaled, decimated_meshes = scale_and_decimate_mesh(scene, 10, 4.5)

for i, mesh in enumerate(decimated_meshes):
    print(i, len(mesh.faces))

for i, mesh in enumerate(decimated_meshes):
    mesh.export(OUTPUT_PATH / f"mesh_lod{i}.glb")

new_scene = trimesh.Scene()
for i, mesh in enumerate(decimated_meshes):
    new_scene.add_geometry(mesh.copy().apply_translation([i * 20, 0, 0]))


new_scene.export(OUTPUT_PATH / "meshoutput.glb")

# new_scene.show()

min_lod_mesh = decimated_meshes[0]
max_lod_mesh = decimated_meshes[-1]
print(len(decimated_meshes))

generate_standalone_sharded_multiresolution_mesh(
    trimesh.Scene(min_lod_mesh), OUTPUT_PATH / "min_lod_mesh", 0
)
generate_standalone_sharded_multiresolution_mesh(
    trimesh.Scene(max_lod_mesh), OUTPUT_PATH / "max_lod_mesh", 0
)
