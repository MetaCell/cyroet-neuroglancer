import argparse
import json
from pathlib import Path
from typing import Any

import ndjson
import neuroglancer
import neuroglancer.cli
import neuroglancer.static_file_server
from neuroglancer import AnnotationLayer, AnnotationPropertySpec, CoordinateSpace
from neuroglancer.server import sys
from neuroglancer.write_annotations import AnnotationWriter

from cryo_et_neuroglancer.sharding import ShardingSpecification, jsonify


def load_data(
    metadata_path: Path, annotations_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load in the metadata (json) and annotations (ndjson) files."""
    with open(metadata_path, mode="r") as f:
        metadata = json.load(f)
    with open(annotations_path, mode="r") as f:
        annotations = ndjson.load(f)
    return metadata, annotations


def build_rotation_matrix_propertie() -> list[AnnotationPropertySpec]:
    return [
        AnnotationPropertySpec(id=f"rot_mat_{i}_{j}", type="float32")
        for i in range(3)
        for j in range(3)
    ]


def write_annotations(
    output_dir: Path,
    annotations: tuple[dict[str, Any], list[dict[str, Any]]],
    coordinate_space: CoordinateSpace,
    color: tuple[int, int, int, int],
) -> Path:
    """
    Create a neuroglancer annotation folder with the given annotations.

    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    metadata, data = annotations
    name = metadata["annotation_object"]["name"]

    is_oriented = data[0]["type"] == "orientedPoint"
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            AnnotationPropertySpec(id="size", type="float32"),
            AnnotationPropertySpec(id="point_color", type="rgba"),
            AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=[
                    0,
                ],
                enum_labels=[
                    name,
                ],
            ),
            # Spec must be added at the object construction time, not after
            *(build_rotation_matrix_propertie() if is_oriented else []),
        ],
    )

    size = metadata["annotation_object"].get("diameter", 1)
    # TODO not sure what units the diameter is in
    size = size / 100
    for index, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        rot_mat = {
            f"rot_mat_{i}_{j}": col
            for i, line in enumerate(p["xyz_rotation_matrix"])
            for j, col in enumerate(line)
        }
        writer.add_point(location, size=size, point_color=color, name=name, **rot_mat)

    writer.write(output_dir)

    return output_dir


def _shard_by_id_index(directory: Path, shard_bits: int, minishard_bits: int):
    sharding_specification = ShardingSpecification(
        type="neuroglancer_uint64_sharded_v1",
        preshift_bits=0,
        hash="identity",
        minishard_bits=minishard_bits,
        shard_bits=shard_bits,
        minishard_index_encoding="gzip",
        data_encoding="gzip",
    )
    labels = {}
    for file in (directory / "by_id").iterdir():
        if ".shard" not in file.name:
            labels[int(file.name)] = file.read_bytes()
            file.unlink()

    shard_files = sharding_specification.synthesize_shards(labels, progress=True)
    for shard_filename, shard_content in shard_files.items():
        (directory / "by_id" / shard_filename).write_bytes(shard_content)

    info_path = directory / "info"
    info = json.load(info_path.open("r", encoding="utf-8"))
    info["by_id"]["sharding"] = sharding_specification.to_dict()
    info_path.write_text(jsonify(info, indent=2))


def view_data(coordinate_space: CoordinateSpace, output_dir: Path) -> None:
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()

    # Start a static file server, serve the contents of the output directory.
    server = neuroglancer.static_file_server.StaticFileServer(
        static_dir=output_dir,
        bind_address=args.bind_address or "127.0.0.1",
        daemon=True,
    )

    with viewer.txn() as s:
        s.layers["annotations"] = AnnotationLayer(
            source=f"precomputed://{server.url}",
            tab="rendering",
            shader="""
void main() {
    setColor(prop_point_color());
    setPointMarkerSize(prop_size());
}
    """,
        )
        s.selected_layer.layer = "annotations"
        s.selected_layer.visible = True
        s.show_slices = False

        s.dimensions = coordinate_space

    print(viewer)


# TODO support hex colors
def main(
    json_path: Path,
    output: Path,
    resolution: float,
    color: tuple[int, int, int, int],
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    """For each path set, load the data and write the combined annotations."""
    if shard_by_id and len(shard_by_id) < 2:
        shard_by_id = (0, 10)

    annotations = load_data(json_path, json_path.with_suffix(".ndjson"))
    if len(annotations) == 0:
        print(f"No annotation found in {json_path.with_suffix('.ndjson')!s}")
        sys.exit(-1)

    coordinate_space = CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[resolution, resolution, resolution],
    )
    write_annotations(output, annotations, coordinate_space, color)
    print("Wrote annotations to", output)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output, shard_bits, minishard_bits)
