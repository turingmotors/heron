from typing import Optional, Union

import datasets
import pandas as pd
import yaml
from torch.utils.data import Dataset

from ..models.prepare_processors import get_processor
from .ja_csv_datasets import JapaneseCSVDataset
from .m3it_datasets import M3ITDataset


def get_dataset(config: dict) -> Union[Dataset, Dataset]:
    with open(config["dataset_config_path"], "r") as f:
        dataset_config = yaml.safe_load(f)

    processor = get_processor(
        config["settings"]["model_name"], config["settings"]["vision_model_name"]
    )

    if dataset_config.get("dataset_type") == "m3it":
        train_dataset = M3ITDataset.create(dataset_config, processor, "train")
        val_dataset = M3ITDataset.create(dataset_config, processor, "validation")
    elif dataset_config.get("dataset_type") == "japanese_csv":
        train_dataset = JapaneseCSVDataset.create(dataset_config, processor, "train")
        val_dataset = JapaneseCSVDataset.create(dataset_config, processor, "validation")
    else:
        raise ValueError(f"dataset_type: {config.get('dataset_type')} is not supported.")

    return train_dataset, val_dataset
