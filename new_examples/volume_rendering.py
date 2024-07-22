import matplotlib.pyplot as plt

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.volume_rendering import volume_render
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    ContrastLimitCalculator,
)
import numpy as np

FILEPATH = r"/media/starfish/LargeSSD/data/cryoET/data/0004_image/Tomograms/VoxelSpacing13.48/CanonicalTomogram/TS_0004.zarr"

dask_data = load_omezarr_data(FILEPATH, resolution_level=2)
data_array = dask_data.compute()
contrast_calculator = ContrastLimitCalculator(data_array)
contrast_calculator.set_volume_and_z_limits(data_array, central_z_slice=10, z_radius=5)
contrast_limits = contrast_calculator.contrast_limits_from_percentiles(0.0, 100.0)
print(contrast_limits)

rendered_mem, image_shape = volume_render(
    data_array, contrast_limits=contrast_limits, depth_samples=256
)
volume_rendered_image = np.ndarray(
    image_shape, dtype=np.float32, buffer=rendered_mem.buf
)

try:
    # Save the RGBA image
    print("Saving the volume rendered image to volume_rendered_image.png")
    plt.imsave("volume_rendered_image.png", volume_rendered_image)
    del volume_rendered_image
except Exception as e:
    print(f"Error saving the volume rendered image: {e}")
finally:
    rendered_mem.close()
    rendered_mem.unlink()
