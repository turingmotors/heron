#!/bin/bash

# Get the absolute path of the script itself
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the directory two levels up
PARENT_DIR="$(dirname "$(dirname "$(dirname "$DIR")")")"

# Create target dataset directory
PATH_TO_DATA_DIR="${PARENT_DIR}/data/coco"
mkdir -p $PATH_TO_DATA_DIR

# download coco from http://cocodataset.org/#download
wget -P $PATH_TO_DATA_DIR http://images.cocodataset.org/zips/train2014.zip
wget -P $PATH_TO_DATA_DIR http://images.cocodataset.org/zips/val2014.zip
wget -P $PATH_TO_DATA_DIR http://images.cocodataset.org/zips/test2014.zip

# unzip image data
unzip $PATH_TO_DATA_DIR/train2014.zip -d $PATH_TO_DATA_DIR
unzip $PATH_TO_DATA_DIR/val2014.zip -d $PATH_TO_DATA_DIR
unzip $PATH_TO_DATA_DIR/test2014.zip -d $PATH_TO_DATA_DIR

# delete zip files
rm $PATH_TO_DATA_DIR/train2014.zip
rm $PATH_TO_DATA_DIR/val2014.zip
rm $PATH_TO_DATA_DIR/test2014.zip

# dwonload STAIR captions' annotation
wget -P $PATH_TO_DATA_DIR https://raw.githubusercontent.com/STAIR-Lab-CIT/STAIR-captions/master/stair_captions_v1.2.tar.gz
tar -zxvf $PATH_TO_DATA_DIR/stair_captions_v1.2.tar.gz -C $PATH_TO_DATA_DIR

# delete tar.gz files
rm $PATH_TO_DATA_DIR/stair_captions_v1.2.tar.gz

# make csv files for training
python3 $PARENT_DIR/heron/datasets/preprocess/make_STAIR_csv.py \
    --path_to_coco $PATH_TO_DATA_DIR
