#!/bin/bash

REMOTE_DATA_STORE_URL="http://127.0.0.1:9000"
LOCAL_DATA_STORE_PATH="/media/starfish/LargeSSD/data/cryoET/data"

RELATIVE_TOMOGRAM_PATH="10000_TS_26/CanonicalTomogram/TS_026.zarr"
RELATIVE_FA_ANN_PATH="10000_TS_26/Annotations/sara_goetz-fatty_acid_synthase-1.0.json"
RELATIVE_RIBO_ANN_PATH="10000_TS_26/Annotations/sara_goetz-ribosome-1.0.json"

cryoet-converter create-image $RELATIVE_TOMOGRAM_PATH -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_TOMOGRAM_PATH" -u $REMOTE_DATA_STORE_URL -r 1.348 -n "TS_026" -o "$LOCAL_DATA_STORE_PATH/TS_026_tomogram.json"

cryoet-converter encode-annotation "$LOCAL_DATA_STORE_PATH/$RELATIVE_RIBO_ANN_PATH" -o "$LOCAL_DATA_STORE_PATH/TS_026_ribosome" -r 1.348 -c "#ff0000" --shard-by-id 1
cryoet-converter create-annotation TS_026_ribosome -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_FA_ANN_PATH" -u $REMOTE_DATA_STORE_URL -n "TS_026_ribosome" -o "$LOCAL_DATA_STORE_PATH/TS_026_ribosome.json" -c "#ff0000 red" -s 0.2

cryoet-converter encode-annotation "$LOCAL_DATA_STORE_PATH/$RELATIVE_FA_ANN_PATH" -o "$LOCAL_DATA_STORE_PATH/TS_026_fatty_acid" -r 1.348 -c "#0000ff" --shard-by-id 1
cryoet-converter create-annotation TS_026_fatty_acid -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_FA_ANN_PATH" -u $REMOTE_DATA_STORE_URL -n "TS_026_fatty_acid" -o "$LOCAL_DATA_STORE_PATH/TS_026_fatty_acid.json" -c "#0000ff blue" -s 0.2

cryoet-converter combine-json "$LOCAL_DATA_STORE_PATH/TS_026_tomogram.json" "$LOCAL_DATA_STORE_PATH/TS_026_ribosome.json" "$LOCAL_DATA_STORE_PATH/TS_026_fatty_acid.json" -o "$LOCAL_DATA_STORE_PATH/TS_026.json" -r 1.348
cryoet-converter load-state "$LOCAL_DATA_STORE_PATH/TS_026.json"