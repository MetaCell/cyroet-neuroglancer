import argparse
import json
from pathlib import Path
from typing import Any

import ndjson
import neuroglancer
import neuroglancer.cli
import neuroglancer.static_file_server
import neuroglancer.write_annotations


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
    annotations: list[dict[str, Any]],
    coordinate_space: neuroglancer.CoordinateSpace,
    colors: list[tuple[int, int, int, int]],
) -> None:
    """
    Create a neuroglancer annotation folder with the given annotations.

    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    names = [a["metadata"]["annotation_object"]["name"] for a in annotations]
    writer = neuroglancer.write_annotations.AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
            neuroglancer.AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=[i for i in range(len(names))],
                enum_labels=names,
            ),
        ],
    )

    for i, a in enumerate(annotations):
        data = a["data"]
        size = a["metadata"]["annotation_object"]["diameter"]
        # TODO not sure what units the diameter is in
        size = size / 1000
        name = i
        color = colors[i]
        for p in data:
            location = [p["location"][k] for k in ["x", "y", "z"]]
            writer.add_point(location, size=size, point_color=color, name=name)

    writer.write(output_dir)


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


def main(zarr_paths: list[Path], output: Path, resolution: float) -> None:
    """For each path set, load the data and write the combined annotations."""
    annotations = []
    for path_set in zarr_paths:
        input_metadata_path = path_set
        input_annotations_path = path_set.parent / (path_set.stem + ".ndjson")
        metadata, data = load_data(input_metadata_path, input_annotations_path)
        annotations.append({"metadata": metadata, "data": data})

    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[resolution, resolution, resolution],
    )
    colors = [(255, 0, 0, 255), (0, 255, 0, 255)]
    write_annotations(output, annotations, coordinate_space, colors)
    print("Wrote annotations to", output)
