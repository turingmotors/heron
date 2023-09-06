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


import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .base_datasets import BaseDataset

HFProcessor = "HFProcessor"


class JapaneseCSVDataset(BaseDataset):
    """Dataset for Custom Japanese CSV V&L Dataset learning"""

    def __init__(
        self,
        loaded_dataset: pd.DataFrame,
        processor: HFProcessor,
        max_length: int,
        dataset_root: str,
        is_inference: bool = False,
    ):
        super(JapaneseCSVDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.unique_img_path = loaded_dataset.img_path.unique()

        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference
        self.dataset_root = dataset_root

    @classmethod
    def create(
        cls,
        dataset_config: dict,
        processor: HFProcessor,
        max_length: int,
        split: str = "train",
        is_inference: bool = False,
    ):
        dataset_root = dataset_config["dataset_root"]
        target_dataset_list = []
        if "coco" in dataset_config["dataset_names"]:
            if split == "train":
                df_train = pd.read_csv(os.path.join(dataset_root, "data/coco/df_train.csv"))
                target_dataset_list.append(df_train)
            else:
                df_val = pd.read_csv(os.path.join(dataset_root, "data/coco/df_val.csv"))
                target_dataset_list.append(df_val)

        if "visual_genome" in dataset_config["dataset_names"]:
            df_vg = pd.read_csv(os.path.join(dataset_root, "data/visual_genome_ja/df_vg.csv"))
            if split != "train":
                val_ratio = 0.1
                num_val = int(len(df_vg) * val_ratio)
                df_vg = df_vg[:num_val]
            target_dataset_list.append(df_vg)
        else:
            raise ValueError(
                f"dataset_type: {dataset_config.get('dataset_type')} is not supported."
            )

        target_dataframe = pd.concat(target_dataset_list, axis=0, ignore_index=True)

        return cls(
            target_dataframe,
            processor,
            max_length,
            is_inference=is_inference,
            dataset_root=dataset_root,
        )

    def __len__(self) -> int:
        return len(self.unique_img_path)

    def preprocess_image(self, image):
        return self.processor(images=[image], return_tensors="pt")["pixel_values"][0]

    def tokenize(self, text):
        if self.is_inference:
            kwargs = {}
        else:
            kwargs = {"padding": "max_length", "max_length": self.max_length, "truncation": True}
        return self.processor.tokenizer(text=text, return_tensors="pt", **kwargs)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        img_path = self.unique_img_path[index]

        df_interest = self.loaded_dataset[self.loaded_dataset.img_path == img_path].reset_index(
            drop=True
        )
        text = ""

        # concatenate text data
        order = list(range(len(df_interest)))
        random.shuffle(order)
        for i in order:
            row = df_interest.iloc[i]
            question = row["question"]  # str
            answer = row["caption"]  # str
            text += f"##human: {question}\n##gpt: {answer}\n"

        # remove final space
        text = text[: len(text) - 1]

        # imageのロード
        image = Image.open(os.path.join(self.dataset_root, img_path)).convert("RGB")
        img = np.array(image)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        inputs = self.processor(
            images=img,
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # batch size 1 -> unbatch
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _get_item_inference(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        img_path = self.unique_img_path[index]

        df_interest = self.loaded_dataset[self.loaded_dataset.img_path == img_path].reset_index(
            drop=True
        )
        text = ""

        row = df_interest.iloc[0]
        question = row["question"]  # str
        answer = row["caption"]  # str
        text += f"##human: {question}\n##gpt: "

        # imageのロード
        img = Image.open(os.path.join(self.dataset_root, img_path)).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        inputs = self.processor(
            text,
            img,
            return_tensors="pt",
        )

        inputs["labels"] = None
        return inputs, img, answer
