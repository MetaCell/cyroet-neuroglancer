#!/bin/bash

REMOTE_DATA_STORE_URL="http://127.0.0.1:9000"
LOCAL_DATA_STORE_PATH="/media/starfish/LargeSSD/data/cryoET/data"

RELATIVE_TOMOGRAM_PATH="00004_sq_df_sorted_zarr/00004_sq_df_sorted"
RELATIVE_MT_SEG_PATH="00004_MT_ground_truth_zarr"
RELATIVE_ACTIN_SEG_PATH="00004_actin_ground_truth_zarr"

cryoet-converter create-image $RELATIVE_TOMOGRAM_PATH -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_TOMOGRAM_PATH" -u $REMOTE_DATA_STORE_URL -r 1.348 -n "0004 tomogram" -o "$LOCAL_DATA_STORE_PATH/00004_tomogram.json"

cryoet-converter encode-segmentation "$LOCAL_DATA_STORE_PATH/$RELATIVE_MT_SEG_PATH" -o "$LOCAL_DATA_STORE_PATH/00004_MT" -r 1.348 --convert-non-zero
cryoet-converter create-segmentation "00004_MT" -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_MT_SEG_PATH" -u $REMOTE_DATA_STORE_URL -n "00004 MT" -o "$LOCAL_DATA_STORE_PATH/00004_MT.json" -c "#ff0000 red"

cryoet-converter encode-segmentation "$LOCAL_DATA_STORE_PATH/$RELATIVE_ACTIN_SEG_PATH" -o "$LOCAL_DATA_STORE_PATH/00004_actin" -r 1.348 --convert-non-zero
cryoet-converter create-segmentation "00004_actin" -z "$LOCAL_DATA_STORE_PATH/$RELATIVE_ACTIN_SEG_PATH" -u $REMOTE_DATA_STORE_URL -n "00004 actin" -o "$LOCAL_DATA_STORE_PATH/00004_actin.json" -c "#0000ff blue"

cryoet-converter combine-json "$LOCAL_DATA_STORE_PATH/00004_tomogram.json" "$LOCAL_DATA_STORE_PATH/00004_MT.json" "$LOCAL_DATA_STORE_PATH/00004_actin.json" -o "$LOCAL_DATA_STORE_PATH/00004.json" -r 1.348
cryoet-converter load-state "$LOCAL_DATA_STORE_PATH/00004.json"