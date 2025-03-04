from cloudvolume import CloudVolume

FILEPATH = (
    r"file:///media/starfish/LargeSSD/data/cryoET/data/new_actin_converted_mesh_no_res/"
)

cv = CloudVolume(FILEPATH)
cv.viewer(port=1031)
