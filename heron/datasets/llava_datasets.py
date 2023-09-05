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
from typing import Dict

from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset
from PIL import Image

from .base_datasets import BaseDataset

HFProcessor = "HFProcessor"


class LlavaDataset(BaseDataset):
    """Dataset for LLaVA"""

    def __init__(
        self,
        loaded_dataset: HFDataset,
        processor: HFProcessor,
        max_length: int,
        is_inference: bool,
        language: str,
        dataset_root: str
    ):
        super(LlavaDataset, self).__init__(is_inference)
        assert language in ["ja", "en"], "given language is not supported"
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference
        self.language = language
        self.dataset_root = dataset_root

    @classmethod
    def create(
        cls,
        dataset_config: Dict,
        processor: HFProcessor,
        max_length: int,
        split: str = "train",
        is_inference: bool = False
    ):
        """
        Args:
            dataset_config: dataset configuration
            processor: HuggingFace Processor class
            split: data split. "train" or "val"
            is_inference: inference mode or not
            language: "ja" or "en".
        """
        hf_dataset = load_dataset("turing-motors/LLaVA-Instruct-150K-JA")
        split_datasets = hf_dataset["train"].train_test_split(test_size=0.05, seed=11)

        if split == "train":
            return cls(
                split_datasets["train"],
                processor,
                max_length,
                is_inference,
                dataset_config["language"],
                dataset_config["dataset_root"],
            )
        
        elif split == "validation":
            return cls(
                split_datasets["test"],
                processor,
                max_length,
                is_inference,
                dataset_config["language"],
                dataset_config["dataset_root"],
            )
        else:
            raise ValueError("given split is invalid")

    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"][0]

    def tokenize(self, text):
        if self.is_inference:
            kwargs = {}
        else:
            kwargs = {"padding": "max_length", "max_length": self.max_length, "truncation": True}
        return self.processor.tokenizer(text=text, return_tensors="pt", **kwargs)

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def get_message(self, c):
        if self.language == "ja":
            message = c["jp"]
        elif self.language == "en":
            message = c["value"]
        else:
            raise ValueError("invalid language")
        return message

    def _get_item_train(self, index):
        row = self.loaded_dataset[index]

        image_path = os.path.join(self.dataset_root, "coco/train2014/COCO_train2014_" + row["image"])
        images = [Image.open(image_path)]

        prompt = ""
        for c in row["conversations"]:
            agent = c["from"]
            message = self.get_message(c)
            prompt += f"##{agent}: {message}\n"

        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        prompt_attn_mask = tokenized["attention_mask"][0]

        return_dict = {
            "input_ids": tokenized_prompt,
            "labels": tokenized_prompt,
            "attention_mask": prompt_attn_mask,
            "pixel_values": self.preprocess_image(images),
        }
        return return_dict

    def _get_item_inference(self, index):
        row = self.loaded_dataset[index]

        image_path = os.path.join(self.dataset_root, "coco/train2014/COCO_train2014_" + row["image"])
        images = [Image.open(image_path)]

        message = self.get_message(row["conversations"][0])
        prompt = f"###human: {message}"

        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        prompt_attn_mask = tokenized["attention_mask"][0]

        return_dict = {
            "input_ids": tokenized_prompt,
            "labels": tokenized_prompt,
            "attention_mask": prompt_attn_mask,
            "pixel_values": self.preprocess_image(images),
            "image": images[0],
            "conversations": row["conversations"],
            "prompt": prompt,
        }
        return return_dict
