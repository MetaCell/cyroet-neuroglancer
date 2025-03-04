from cloudvolume import CloudVolume

FILEPATH = r"file:///media/starfish/LargeSSD/data/cryoET/data/new_MT_converted_mesh/"

cv = CloudVolume(FILEPATH)
cv.viewer(port=1030)
