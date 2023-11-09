from pathlib import Path
from typing import Any, Iterator
from tqdm import tqdm
import numpy as np
import dask.array as da
from cryo_et_neuroglancer.chunk import Chunk
import sys
import shutil

from cryo_et_neuroglancer.utils import iterate_chunks
from cryo_et_neuroglancer.segmentation_encoding import (
    create_segmentation_chunk,
)
from cryo_et_neuroglancer.io import load_omezarr_data, write_metadata


def _create_metadata(
    chunk_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
    data_size: tuple[int, int, int],
    data_directory: str,
) -> dict[str, Any]:
    """Create the metadata for the segmentation"""
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [list(chunk_size)],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": list(block_size),
                # TODO resolution is in nm, while for others there is no units
                "resolution": [1, 1, 1],
                "key": data_directory,
                "size": data_size[::-1]  # reverse the data size to pass from X-Y-Z to Z-Y-X
            }
        ],
        "type": "segmentation",
    }
    return metadata


def create_segmentation(
    dask_data: da.Array, block_size: tuple[int, int, int]
) -> Iterator[Chunk]:
    """Yield the neuroglancer segmentation format chunks"""
    to_iterate = iterate_chunks(dask_data)
    num_iters = np.prod(dask_data.numblocks)
    for chunk, dimensions in tqdm(
        to_iterate, desc="Processing chunks", total=num_iters
    ):
        yield create_segmentation_chunk(chunk.compute(), dimensions, block_size)


def main(
    filename: Path,
    block_size: tuple[int, int, int] = (64, 64, 64),
    data_directory: str = "data",
    delete_existing_output_directory: bool = False
) -> None:
    """Convert the given OME-Zarr file to neuroglancer segmentation format with the given block size"""
    print(f"Converting {filename} to neuroglancer compressed segmentation format")
    dask_data = load_omezarr_data(filename)
    output_directory = filename.parent / f"precomputed-{filename.stem[:-5]}"
    if delete_existing_output_directory and output_directory.exists():
        print(f"The output directory {output_directory!s} exists, deleting before starting the conversion")
        shutil.rmtree(output_directory)
    elif not delete_existing_output_directory and output_directory.exists():
        print(f"The output directory {output_directory!s} already exists")
        sys.exit(1)
    output_directory.mkdir(parents=True, exist_ok=True)
    for c in create_segmentation(dask_data, block_size):
        c.write_to_directory(output_directory / data_directory)


    metadata = _create_metadata(
        dask_data.chunksize, block_size, dask_data.shape, data_directory
    )
    write_metadata(metadata, output_directory)
    print(f"Wrote segmentation to {output_directory}")


if __name__ == "__main__":
    # TODO create command line interface
    if len(sys.argv) < 2:
        print("Missing argument (folder)")
        sys.exit(-1)
    actin_file_path = Path(sys.argv[1])
    if not actin_file_path.exists():
        print("The data folder doesn't exist")
        sys.exit(-2)
    block_size = (32, 32, 32)
    main(actin_file_path, block_size, delete_existing_output_directory=True)
