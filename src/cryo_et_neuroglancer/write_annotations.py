import json
from pathlib import Path
from typing import Any

import ndjson
from neuroglancer import AnnotationPropertySpec, CoordinateSpace
from neuroglancer.server import sys
from neuroglancer.write_annotations import AnnotationWriter

from cryo_et_neuroglancer.sharding import ShardingSpecification, jsonify
from cryo_et_neuroglancer.utils import parse_color


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
            AnnotationPropertySpec(id="diameter", type="float32"),
            AnnotationPropertySpec(id="point_color", type="rgba"),
            AnnotationPropertySpec(id="point_index", type="float32"),
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

    # Convert angstrom to nanometer
    # Using 28nm as default size
    diameter = metadata["annotation_object"].get("diameter", 280) / 10
    for index, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        if is_oriented:
            rot_mat = {
                f"rot_mat_{i}_{j}": col
                for i, line in enumerate(p["xyz_rotation_matrix"])
                for j, col in enumerate(line)
            }
        else:
            rot_mat = {}
        writer.add_point(
            location,
            diameter=diameter,
            point_color=color,
            point_index=float(index),
            name=0,
            **rot_mat,
        )

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


def main(
    json_path: Path,
    output: Path,
    resolution: float,
    color: list[str],
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    """For each path set, load the data and write the combined annotations."""
    annotations = load_data(json_path, json_path.with_suffix(".ndjson"))
    if len(annotations) == 0:
        print(f"No annotation found in {json_path.with_suffix('.ndjson')!s}")
        sys.exit(-1)
    process_annotation(annotations, output, resolution, color, shard_by_id)


def process_annotation(
    annotations: tuple[dict[str, Any], list[dict[str, Any]]],
    output: Path,
    resolution: float,
    color: list[str],
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    if shard_by_id and len(shard_by_id) < 2:
        shard_by_id = (0, 10)

    parsed_color = parse_color(color)

    coordinate_space = CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[resolution, resolution, resolution],
    )
    write_annotations(output, annotations, coordinate_space, parsed_color)
    print("Wrote annotations to", output)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output, shard_bits, minishard_bits)
