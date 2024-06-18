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
    r"/media/starfish/LargeSSD/data/cryoET/data/oriented/liang_xue-chloramphenicol_bound_70s_ribosome-1.0.json"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/new_oriented_points/")

metadata, data = load_data(JSON_PATH, JSON_PATH.with_suffix(".ndjson"))

encode_annotation(
    data, metadata, OUTPUT_PATH, 1.0, shard_by_id=(0, 10), is_oriented=True
)

SOURCE = "http://127.0.0.1:9000/new_oriented_points/"

output = generate_oriented_point_layer(
    source=SOURCE,
    name="Test Oriented Points",
    color="#FF0000",
    point_size_multiplier=1.0,
    is_visible=True,
    is_instance_segmentation=False,
)
print(output)

layer_json = combine_json_layers([output], 1.0)


json.dump(layer_json, open("oriented_point_layer.json", "w"), indent=2)
