#!/bin/bash

REMOTE_DATA_STORE_URL="http://127.0.0.1:9000"
LOCAL_DATA_STORE_PATH="/media/starfish/LargeSSD/data/cryoET/data"

RELATIVE_ANN_PATH="oriented/liang_xue-chloramphenicol_bound_70s_ribosome-1.0.json"

cryoet-converter encode-annotation "$LOCAL_DATA_STORE_PATH/$RELATIVE_ANN_PATH" -o "$LOCAL_DATA_STORE_PATH/oriented_ribosome" -r 1.348 -c "#00b3b3" --shard-by-id 1
cryoet-converter create-annotation oriented_ribosome -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_ANN_PATH" -u $REMOTE_DATA_STORE_URL -n "oriented_ribosome" -o "$LOCAL_DATA_STORE_PATH/oriented_ribosome.json" -c "#ff0000 red" -s 0.1

cryoet-converter combine-json "$LOCAL_DATA_STORE_PATH/oriented_ribosome.json" -o "$LOCAL_DATA_STORE_PATH/oriented_ribosome_state.json" -r 1.348
cryoet-converter load-state "$LOCAL_DATA_STORE_PATH/oriented_ribosome_state.json"