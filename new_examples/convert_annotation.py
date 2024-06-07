import json

import ndjson
from typing import Any
from cryoet_data_portal_neuroglancer.precompute.points import (
    encode_annotation,
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
    r"/media/starfish/LargeSSD/data/cryoET/data/10000_TS_26/Annotations/sara_goetz-fatty_acid_synthase-1.0.json"
)

OUTPUT_PATH = Path(r"/media/starfish/LargeSSD/data/cryoET/data/new_fatty_acid_synthase")

metadata, data = load_data(JSON_PATH, JSON_PATH.with_suffix(".ndjson"))

encode_annotation(data, metadata, OUTPUT_PATH, 1.0, shard_by_id=None)
