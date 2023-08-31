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


from typing import Tuple, Dict

import datasets
import pandas as pd

import yaml
from torch.utils.data import Dataset

from ..models.prepare_processors import get_processor
from .ja_csv_datasets import JapaneseCSVDataset
from .m3it_datasets import M3ITDataset
from .llava_datasets import LlavaDataset


def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    
    with open(config["dataset_config_path"], "r") as f:
        dataset_config = yaml.safe_load(f)

    processor = get_processor(config["model_config"])

    if dataset_config.get("dataset_type") == "m3it":
        train_dataset = M3ITDataset.create(dataset_config, processor, "train")
        val_dataset = M3ITDataset.create(dataset_config, processor, "validation")
    
    elif dataset_config.get("dataset_type") == "japanese_csv":
        train_dataset = JapaneseCSVDataset.create(dataset_config, processor, "train")
        val_dataset = JapaneseCSVDataset.create(dataset_config, processor, "validation")
    
    elif dataset_config.get("dataset_type") == "llava":
        train_dataset = LlavaDataset.create(dataset_config, processor, "train")
        val_dataset = LlavaDataset.create(dataset_config, processor, "validation")
    
    else:
        raise ValueError(f"dataset_type: {config.get('dataset_type')} is not supported.")

    return train_dataset, val_dataset
