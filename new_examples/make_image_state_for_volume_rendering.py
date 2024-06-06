import json

from cryoet_data_portal_neuroglancer.state_generator import (
    generate_image_volume_layer,
    generate_image_layer,
    combine_json_layers,
)

# Point to some local zarr data - here served by a CORS webserver
SOURCE = "http://127.0.0.1:9000/00004_sq_df_sorted_zarr/00004_sq_df_sorted"

layer_img_json = generate_image_layer(
    source=SOURCE,
    scale=(1.0, 1.0, 1.0),
    size={"x": 1.0, "y": 1.0, "z": 1.0},
    name="Test VR",
)
print(layer_img_json)

layer_vr_json = generate_image_volume_layer(
    source=SOURCE,
    name="Test Volume Rendering",
)

json_output = combine_json_layers([layer_vr_json], 1.0)

print(json_output)

json.dump(json_output, open("image_volume_layer.json", "w"), indent=2)
