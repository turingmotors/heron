#!/bin/bash

PATH_TO_DATA_DIR="./data/visual_genome_ja"
mkdir -p $PATH_TO_DATA_DIR

# download coco from http://cocodataset.org/#download
wget -P $PATH_TO_DATA_DIR https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P $PATH_TO_DATA_DIR https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# unzip image data
unzip $PATH_TO_DATA_DIR/images.zip -d $PATH_TO_DATA_DIR
unzip $PATH_TO_DATA_DIR/images2.zip -d $PATH_TO_DATA_DIR

# delete zip files
rm $PATH_TO_DATA_DIR/images.zip
rm $PATH_TO_DATA_DIR/images2.zip

# move images2 to images
mv $PATH_TO_DATA_DIR/VG_100K_2/* $PATH_TO_DATA_DIR/VG_100K/
rm -rf $PATH_TO_DATA_DIR/VG_100K_2

# dwonload STAIR captions' annotation
wget -P $PATH_TO_DATA_DIR https://raw.githubusercontent.com/yahoojapan/ja-vg-vqa/master/question_answers.json.zip
unzip $PATH_TO_DATA_DIR/question_answers.json.zip -d $PATH_TO_DATA_DIR

# delete tar.gz files
rm $PATH_TO_DATA_DIR/question_answers.json.zip

# make csv files for training
python3 ./heron/datasets/preprocess/make_japanese_visual_genome_csv.py \
    --path_to_visual_genome $PATH_TO_DATA_DIR
