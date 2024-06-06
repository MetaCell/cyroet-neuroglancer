import json

from cryoet_data_portal_neuroglancer.state_generator import (
    generate_image_volume_layer,
    combine_json_layers,
)

# Point to some local zarr data - here served by a CORS webserver
SOURCE = "http://127.0.0.1:9000/00004_sq_df_sorted_zarr/00004_sq_df_sorted"

layer_json = generate_image_volume_layer(
    source=SOURCE,
    name="Test VR",
)

json_output = combine_json_layers([layer_json], 1.0)

print(json_output)

json.dump(json_output, open("image_volume_layer.json", "w"), indent=2)
