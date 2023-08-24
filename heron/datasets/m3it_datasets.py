from base64 import b64decode
from io import BytesIO

import cv2
import datasets
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlamaTokenizer,
)


class M3ITDataset(Dataset):
    """Dataset for M3IT Dataset learning
    """
    def __init__(
        self,
        model_name: str,
        vision_model_name: str,
        loaded_dataset: datasets.GeneratorBasedBuilder,
        max_length: int = 128,
    ):
        super(M3ITDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        if "japanese-stablelm" in model_name:
            self.processor.tokenizer = LlamaTokenizer.from_pretrained(
                "novelai/nerdstash-tokenizer-v1",
                padding_side="right",
                additional_special_tokens=["▁▁"],
            )
        elif (
            "mpt" in model_name
            or "matsuo-lab/weblab" in model_name
            or "cyberagent/open-calm-7b" in model_name
        ):
            self.processor.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="right", use_fast=True
            )
        else:
            self.processor.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="right", use_fast=False
            )
        if "llama" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        elif "mpt" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        elif "matsuo-lab/weblab" in model_name:
            self.processor.tokenizer.add_special_tokens(
                {
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "pad_token": "<|padding|>",
                    "unk_token": "<|endoftext|>",
                }
            )

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def __getitem__(self, index) -> dict:
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset[index]

        # some of nlvr data were broken
        instruction = row["instruction"]  # str
        question = row["inputs"]  # str
        answer = row["outputs"]  # str
        text = f"##Instruction: {instruction} ##Question: {question} ##Answer: {answer}"

        # imageのロード
        image_base64_str_list = row["image_base64_str"]  # str (base64)
        img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        inputs = self.processor(
            text,
            img,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # batch size 1 -> unbatch
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

