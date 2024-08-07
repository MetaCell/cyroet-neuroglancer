import itertools
import matplotlib.pyplot as plt

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.volume_rendering import volume_render
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    ContrastLimitCalculator,
)
import numpy as np

FILEPATH = r"/media/starfish/LargeSSD/data/cryoET/data/0004_image/Tomograms/VoxelSpacing13.48/CanonicalTomogram/TS_0004.zarr"

dask_data = load_omezarr_data(FILEPATH, resolution_level=1)
data_array = dask_data.compute()

# def rms(data):
#     return np.sqrt(np.mean(data**2))


# standard_deviation_per_z_slice = np.std(data_array, axis=(1, 2))
# standard_deviation_per_z_slice = np.nan_to_num(
#     standard_deviation_per_z_slice, copy=False
# )
# for i, std in enumerate(standard_deviation_per_z_slice):
#     print(f"Standard deviation for z-slice {i}: {std:.2f}")

# lowest_points = find_peaks(-standard_deviation_per_z_slice, prominence=0.1)
# print(lowest_points)

# fig, ax = plt.subplots()
# ax.plot(standard_deviation_per_z_slice)
# ax.set_xlabel("Z-slice")
# ax.set_ylabel("Standard deviation")
# plt.show()

# exit(-1)

contrast_calculator = ContrastLimitCalculator()

percentile_thresholds = [(0.0, 100.0), (45.0, 99.5), (5.0, 95.0)]
for (low_percentile, high_percentile), clip in itertools.product(
    percentile_thresholds, [True, False]
):
    contrast_calculator.volume = data_array
    if clip:
        contrast_calculator.trim_volume_around_central_zslice()

    contrast_limits = contrast_calculator.contrast_limits_from_percentiles(
        low_percentile, high_percentile
    )

    rendered_mem, image_shape = volume_render(
        data_array, contrast_limits=contrast_limits, depth_samples=256
    )
    volume_rendered_image = np.ndarray(
        image_shape, dtype=np.float32, buffer=rendered_mem.buf
    )

    try:
        # Save the RGBA image
        print(
            f"Saving volume rendered image with contrast limits {contrast_limits} and clipping {clip}"
        )
        clip_string = "clipped" if clip else "unclipped"
        plt.imsave(
            f"volume_rendered_image_{low_percentile}_{high_percentile}_{clip_string}_{contrast_limits[0]:.2f}_{contrast_limits[1]:.2f}.png",
            volume_rendered_image,
        )
        del volume_rendered_image
    except Exception as e:
        print(f"Error saving the volume rendered image: {e}")
    finally:
        rendered_mem.close()
        rendered_mem.unlink()
