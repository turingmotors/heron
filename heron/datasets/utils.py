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
from .ja_csv_instruct_datasets import JapaneseCSVInstructDataset
from .llava_datasets import LlavaDataset
from .llava_instruct_datasets import LlavaInstructDataset
from .m3it_datasets import M3ITDataset
from .m3it_instruct_datasets import M3ITInstructDataset

dataset_classes = {
    "japanese_csv": JapaneseCSVDataset,
    "japanese_csv_instruct": JapaneseCSVInstructDataset,
    "llava": LlavaDataset,
    "llava_instruct": LlavaInstructDataset,
    "m3it": M3ITDataset,
    "m3it_instruct": M3ITInstructDataset,
}


def get_each_dataset(
    dataset_config: Dict, processor, max_length: int, model_type: str
) -> Tuple[Dataset, Dataset]:
    dataset_type = dataset_config["dataset_type"]
    if dataset_type not in dataset_classes:
        raise ValueError(f"dataset_type: {dataset_type} is not supported.")

    DatasetClass = dataset_classes[dataset_type]
    train_dataset = DatasetClass.create(dataset_config, processor, max_length, model_type, "train")
    val_dataset = DatasetClass.create(
        dataset_config, processor, max_length, model_type, "validation"
    )
    return train_dataset, val_dataset


def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    processor = get_processor(config["model_config"])
    train_dataset_list = []
    val_dataset_list = []
    max_length = config["model_config"]["max_length"]
    model_type = config["model_config"]["model_type"]

    for dataset_config_path in config["dataset_config_path"]:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        train_dataset, val_dataset = get_each_dataset(
            dataset_config, processor, max_length, model_type
        )
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    return train_dataset, val_dataset
