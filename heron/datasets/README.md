# Datasets Description

# Supported Datasets

## English
- [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT)

## Japanese CSV
- [STAIR](http://captions.stair.center/)
- [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa)

For preparing csv files, run the following commands;
```bash
# For COCO/STAIR dataset
./heron/datasets/preprocess/download_and_preprocessing_coco.sh

# For Japanese Visual Genome VQA dataset
./heron/datasets/preprocess/download_and_preprocessing_visual_genome.sh
```

Data will be located as following;
```bash
data/
├── coco
│   ├── df_train.csv
│   ├── df_val.csv
│   ├── stair_captions_v1.2_train.json
│   ├── stair_captions_v1.2_train_tokenized.json
│   ├── stair_captions_v1.2_val.json
│   ├── stair_captions_v1.2_val_tokenized.json
│   ├── test2014
│   ├── train2014
│   └── val2014
└── visual_genome_ja
    ├── VG_100K
    ├── df_vg.csv
    └── question_answers.json
```
