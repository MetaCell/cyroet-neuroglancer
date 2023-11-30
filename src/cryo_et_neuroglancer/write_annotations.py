import argparse
import json
from pathlib import Path
from typing import Any

import ndjson
import neuroglancer
import neuroglancer.cli
import neuroglancer.static_file_server
import neuroglancer.write_annotations

from cryo_et_neuroglancer.sharding import ShardingSpecification, jsonify


def load_data(
    metadata_path: Path, annotations_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load in the metadata (json) and annotations (ndjson) files."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(annotations_path) as f:
        annotations = ndjson.load(f)
    return metadata, annotations


def write_annotations(
    output_dir: Path,
    annotations: tuple[dict[str, Any], list[dict[str, Any]]],
    coordinate_space: neuroglancer.CoordinateSpace,
    color: tuple[int, int, int, int],
) -> Path:
    """
    Create a neuroglancer annotation folder with the given annotations.

    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    metadata, data = annotations
    name = metadata["annotation_object"]["name"]
    writer = neuroglancer.write_annotations.AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
            neuroglancer.AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=[
                    0,
                ],
                enum_labels=[
                    name,
                ],
            ),
        ],
    )

    size = metadata["annotation_object"]["diameter"]
    # TODO not sure what units the diameter is in
    size = size / 100
    for i, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        writer.add_point(location, size=size, point_color=color, name=0)

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


def view_data(coordinate_space: neuroglancer.CoordinateSpace, output_dir: Path) -> None:
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
        s.layers["annotations"] = neuroglancer.AnnotationLayer(
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
# TODO handle cases where information is missing
def main(
    json_path: Path,
    output: Path,
    resolution: float,
    color: tuple[int, int, int, int],
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    """For each path set, load the data and write the combined annotations."""
    if len(shard_by_id) < 2:
        shard_by_id = (0, 10)

    annotation = load_data(json_path, json_path.with_suffix(".ndjson"))

    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[resolution, resolution, resolution],
    )
    write_annotations(output, annotation, coordinate_space, color)
    print("Wrote annotations to", output)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output, shard_bits, minishard_bits)
