from pathlib import Path

import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
import numpy as np
from cloudvolume import CloudVolume


def generate_basic_object(output_path, shape=(1024, 1024, 512), radius=0.8):
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    sphere = xx**2 + yy**2 + zz**2 <= radius**2
    sphere = sphere.astype(np.uint32)

    CloudVolume.from_numpy(
        sphere,
        vol_path=output_path,
        resolution=(20, 20, 40),
        chunk_size=(256, 256, 128),
        layer_type="segmentation",
        progress=True,
        compress=False,
    )


def convert(layer_path, num_lod):
    tq = LocalTaskQueue()
    tasks = tc.create_meshing_tasks(
        layer_path,
        mip=0,
        shape=(256, 256, 256),
        sharded=True,
        fill_missing=True,
        max_simplification_error=10,
        simplification=True,
        mesh_dir="mesh",
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(
        layer_path,
        magnitude=3,
        mesh_dir="mesh",
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_sharded_multires_mesh_tasks(
        layer_path,
        num_lod=num_lod,
        min_chunk_size=(32, 32, 16),
        mesh_dir="mesh",
    )
    tq.insert(tasks)
    tq.execute()


def clean_mesh_folder(output_path: str | Path, mesh_directory: str = "mesh"):
    """Remove unnecessary files from the mesh folder

    The conversion produces extra unnecessary files, so we clean up
    The only needed files are the info file, and any file that ends with .shard
    However, if the sharding fails, we may have other files, so we only delete
    If there is at least one shard file
    """
    output_path = Path(output_path)
    mesh_dir = output_path / mesh_directory
    if mesh_dir.exists() and any(f.name.endswith(".shard") for f in mesh_dir.iterdir()):
        for f in mesh_dir.iterdir():
            if f.is_file() and not (f.name == "info" or f.name.endswith(".shard")):
                f.unlink()


def main(output_folder="test_mesh", port=1030, num_lod=4):
    here = Path(__file__).parent
    output_path = here / output_folder
    output_cloudvolume = f"file://{output_path.resolve()}"
    generate_basic_object(output_path=output_cloudvolume)
    convert(output_cloudvolume, num_lod)
    clean_mesh_folder(output_path=output_path)

    cv = CloudVolume(output_cloudvolume)
    cv.viewer(port=port)


if __name__ == "__main__":
    main()
