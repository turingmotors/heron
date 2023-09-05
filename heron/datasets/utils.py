# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, Tuple

import datasets
import pandas as pd
import yaml
from torch.utils.data import ConcatDataset, Dataset

from ..models.prepare_processors import get_processor
from .ja_csv_datasets import JapaneseCSVDataset
from .llava_datasets import LlavaDataset
from .m3it_datasets import M3ITDataset


def get_each_dataset(dataset_config: Dict, processor, max_length: int) -> Tuple[Dataset, Dataset]:
    if dataset_config["dataset_type"] == "m3it":
        train_dataset = M3ITDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = M3ITDataset.create(dataset_config, processor, max_length, "validation")

    elif dataset_config["dataset_type"] == "japanese_csv":
        train_dataset = JapaneseCSVDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = JapaneseCSVDataset.create(
            dataset_config, processor, max_length, "validation"
        )

    elif dataset_config["dataset_type"] == "llava":
        train_dataset = LlavaDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = LlavaDataset.create(dataset_config, processor, max_length, "validation")

    else:
        raise ValueError(f"dataset_type: {dataset_config['dataset_type']} is not supported.")

    return train_dataset, val_dataset


def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    processor = get_processor(config["model_config"])
    train_dataset_list = []
    val_dataset_list = []
    max_length = config["model_config"]["max_length"]

    for dataset_config_path in config["dataset_config_path"]:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        train_dataset, val_dataset = get_each_dataset(dataset_config, processor, max_length)
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    return train_dataset, val_dataset
