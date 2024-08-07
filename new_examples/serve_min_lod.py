from cloudvolume import CloudVolume

FILEPATH = (
    r"file:///media/starfish/LargeSSD/data/cryoET/data/converted-01122021/min_lod_mesh"
)

cv = CloudVolume(FILEPATH)
cv.viewer(port=1031)
