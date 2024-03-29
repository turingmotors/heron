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
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base_datasets import IGNORE_INDEX, BaseDataset

HFProcessor = "HFProcessor"


class JapaneseCSVInstructDataset(BaseDataset):
    """Dataset for Custom Japanese CSV V&L Dataset learning
    This dataset is designed for instruction tuning, meaning it considers the lossese associated with gpt responses.
    """

    def __init__(
        self,
        loaded_dataset: pd.DataFrame,
        processor: HFProcessor,
        max_length: int,
        dataset_root: str,
        is_inference: bool = False,
    ):
        super(JapaneseCSVInstructDataset, self).__init__()
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

    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"][0]

    def tokenize(self, text):
        kwargs = {}
        return self.processor.tokenizer(text=text, return_tensors="pt", **kwargs)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        img_path = self.unique_img_path[index]

        df_interest = self.loaded_dataset[self.loaded_dataset.img_path == img_path].reset_index(
            drop=True
        )
        # imageのロード
        image = Image.open(os.path.join(self.dataset_root, img_path)).convert("RGB")
        image = np.array(image)
        images = [image]

        tokenized_list = []
        labels_list = []
        attn_mask_list = []

        # concatenate text data
        order = list(range(len(df_interest)))
        random.shuffle(order)
        for i, c in enumerate(order):
            if i > 0:
                drop_eos_token = 1
            else:
                drop_eos_token = 0

            row = df_interest.iloc[i]
            question = row["question"]  # str
            answer = row["caption"]  # str
            prompt_q = f"##human: {question}\n##gpt: "
            prompt_a = f"{answer}"

            # ================================
            # tokenize question text
            # ================================
            tokenized = self.tokenize(prompt_q)
            tokenized_prompt = tokenized["input_ids"][0][drop_eos_token:]
            # all label should be ignored
            labels = torch.full_like(tokenized_prompt, IGNORE_INDEX)
            prompt_attn_mask = tokenized["attention_mask"][0][drop_eos_token:]

            tokenized_list.append(tokenized_prompt)
            labels_list.append(labels)
            attn_mask_list.append(prompt_attn_mask)

            # ================================
            # tokenize answer text
            # ================================
            tokenized = self.tokenize(prompt_a)
            tokenized_prompt = tokenized["input_ids"][0][1:]
            # all label should be included in loss
            labels = tokenized_prompt
            prompt_attn_mask = tokenized["attention_mask"][0][1:]

            tokenized_list.append(tokenized_prompt)
            labels_list.append(labels)
            attn_mask_list.append(prompt_attn_mask)

        # =================================================
        # concat question and answer, apply max_length
        # =================================================
        tokenized_prompt = torch.cat(tokenized_list, dim=-1)
        labels = torch.cat(labels_list, dim=-1)
        prompt_attn_mask = torch.cat(attn_mask_list, dim=-1)

        if len(tokenized_prompt) < self.max_length:
            pad_length = self.max_length - len(tokenized_prompt)
            tokenized_prompt = torch.cat(
                [
                    tokenized_prompt,
                    torch.tensor([self.processor.tokenizer.pad_token_id] * pad_length),
                ],
                dim=-1,
            )
            labels = torch.cat([labels, torch.tensor([IGNORE_INDEX] * pad_length)], dim=-1)
            prompt_attn_mask = torch.cat(
                [prompt_attn_mask, torch.tensor([0] * pad_length)], dim=-1
            )
        else:
            tokenized_prompt = tokenized_prompt[: self.max_length]
            labels = labels[: self.max_length]
            prompt_attn_mask = prompt_attn_mask[: self.max_length]

        return_dict = {
            "input_ids": tokenized_prompt,
            "labels": labels,
            "attention_mask": prompt_attn_mask,
            "pixel_values": self.preprocess_image(images),
        }
        return return_dict

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

        inputs = self.processor(
            text,
            img,
            return_tensors="pt",
        )

        inputs["labels"] = None
        return inputs, img, answer
