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


from base64 import b64decode
from io import BytesIO

import cv2
import datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from .base_datasets import IGNORE_INDEX, ResilientDataset

HFProcessor = "HFProcessor"


class M3ITInstructDataset(ResilientDataset):
    """Dataset for M3IT Dataset learning
    This dataset is designed for instruction tuning, meaning it considers the lossese associated with gpt responses.
    """

    def __init__(
        self,
        loaded_dataset: ConcatDataset,
        processor: HFProcessor,
        max_length: int,
        is_inference: bool = False,
    ):
        super(M3ITInstructDataset, self).__init__(is_inference)
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference

    @classmethod
    def create(
        cls,
        dataset_config: dict,
        processor: HFProcessor,
        max_length: int,
        split: str = "train",
        is_inference: bool = False,
    ):
        dataset_list = [
            datasets.load_dataset("MMInstruction/M3IT", i, num_proc=16)
            for i in dataset_config["dataset_names"]
        ]

        # some dataset have no validation
        target_dataset_list = []
        for d in dataset_list:
            try:
                target_dataset_list.append(d[split])
            except KeyError:
                print(f"{d['train']._info.config_name} has no {split} set.")
        target_dataframe = ConcatDataset(target_dataset_list)

        return cls(target_dataframe, processor, max_length, is_inference)

    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"][0]

    def tokenize(self, text):
        kwargs = {}
        return self.processor.tokenizer(text=text, return_tensors="pt", **kwargs)

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset[index]

        # imageのロード
        image_base64_str_list = row["image_base64_str"]  # str (base64)
        image = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        image = np.array(image)
        images = [image]

        tokenized_list = []
        labels_list = []
        attn_mask_list = []

        # some of nlvr data were broken
        instruction = row["instruction"]  # str
        question = row["inputs"]  # str
        answer = row["outputs"]  # str
        prompt_q = f"##human: {instruction} {question}\n##gpt: "
        prompt_a = f"{answer}"

        # ================================
        # tokenize question text
        # ================================
        tokenized = self.tokenize(prompt_q)
        tokenized_prompt = tokenized["input_ids"][0]
        # all label should be ignored
        labels = torch.full_like(tokenized_prompt, IGNORE_INDEX)
        prompt_attn_mask = tokenized["attention_mask"][0]

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
        row = self.loaded_dataset[index]

        # some of nlvr data were broken
        instruction = row["instruction"]  # str
        question = row["inputs"]  # str
        answer = row["outputs"]  # str
        text = f"##Instruction: {instruction} ##Question: {question} ##Answer: "

        # imageのロード
        image_base64_str_list = row["image_base64_str"]  # str (base64)
        image = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        image = np.array(image)
        images = [image]

        inputs = self.processor(
            text,
            images,
            return_tensors="pt",
        )
        inputs["labels"] = None
        return inputs, image, answer
