from cloudvolume import CloudVolume

FILEPATH = (
    r"file:///media/starfish/LargeSSD/data/cryoET/data/converted-01122021/meshoutput/"
)

cv = CloudVolume(FILEPATH)
cv.viewer()
