import json

import ndjson
from typing import Any
from cryoet_data_portal_neuroglancer.precompute.points import (
    encode_annotation,
)
from cryoet_data_portal_neuroglancer.state_generator import (
    generate_oriented_point_layer,
    combine_json_layers,
)
from pathlib import Path


def load_data(
    metadata_path: Path, annotations_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load in the metadata (json) and annotations (ndjson) files."""
    with open(metadata_path, mode="r") as f:
        metadata = json.load(f)
    with open(annotations_path, mode="r") as f:
        annotations = ndjson.load(f)
    return metadata, annotations


JSON_PATH = Path(
    r"/media/starfish/LargeSSD/data/cryoET/data/Annotations-test-run/100-proton_transporting_atp_synthase_complex-1.0.json"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/converted-01122021/")

metadata, data = load_data(
    JSON_PATH, JSON_PATH.with_name(JSON_PATH.stem + "_orientedpoint.ndjson")
)

encode_annotation(
    data, metadata, OUTPUT_PATH, 0.784, shard_by_id=(0, 10), is_oriented=True
)

SOURCE = "http://127.0.0.1:9000/converted-01122021/"

output = generate_oriented_point_layer(
    source=SOURCE,
    name="Test Oriented Points",
    color="#FF0000",
    point_size_multiplier=0.5,
    line_width=2.0,
    is_visible=True,
    is_instance_segmentation=False,
)
layer_json = combine_json_layers([output], 0.784)
json.dump(layer_json, open("oriented_point_layer.json", "w"), indent=2)
