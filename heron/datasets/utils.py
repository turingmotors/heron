from typing import Union, Optional

import datasets
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from .ja_csv_datasets import JapaneseCSVDataset
from .m3it_datasets import M3ITDataset

def get_dataset(config: dict, model_name: str, vision_model_name: str, max_length: int) -> Union[Dataset, Dataset]:
    if config.get("dataset_type") == "m3it":
        dataset_list = [
            datasets.load_dataset("MMInstruction/M3IT", i) for i in config["dataset_names"]
        ]
        train_dataframe = ConcatDataset([d["train"] for d in dataset_list])
        train_dataset = M3ITDataset(model_name, vision_model_name, train_dataframe, max_length)

        # some dataset have no validation
        val_dataset_list = []
        for d in dataset_list:
            try:
                val_dataset_list.append(d["validation"])
            except:
                print(f"{d['train']._info.config_name} has no validation set.")
        val_dataframe = ConcatDataset(val_dataset_list)
        val_dataset = M3ITDataset(model_name, vision_model_name, val_dataframe, max_length)
    elif config.get("dataset_type") == "japanese_csv":
        df_train = pd.read_csv("./data/coco/df_train.csv")
        df_val = pd.read_csv("./data/coco/df_val.csv")
        df_vg = pd.read_csv("./data/visual_genome_ja/df_vg.csv")

        train_dataframe = pd.concat([df_train, df_vg], axis=0, ignore_index=True)
        train_dataset = JapaneseCSVDataset(model_name, vision_model_name, train_dataframe, max_length)

        val_dataframe = df_val
        val_dataset = JapaneseCSVDataset(model_name, vision_model_name, val_dataframe, max_length)
    else:
        raise ValueError(f"dataset_type: {config.get('dataset_type')} is not supported.")

    return train_dataset, val_dataset
