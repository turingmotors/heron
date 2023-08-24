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


class JapaneseCSVDataset(Dataset):
    """Dataset for Custom Japanese CSV V&L Dataset learning
    """

    def __init__(
        self,
        model_name: str,
        vision_model_name: str,
        loaded_dataset: datasets.GeneratorBasedBuilder,
        max_length: int = 128,
    ):
        super(JapaneseCSVDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.unique_img_path = loaded_dataset.img_path.unique()

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
        return len(self.unique_img_path)

    def __getitem__(self, index) -> dict:
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        img_path = self.unique_img_path[index]

        df_interest = self.loaded_dataset[self.loaded_dataset.img_path == img_path].reset_index(
            drop=True
        )
        text = ""

        # concatenate text data
        for i in np.random.randint(0, len(df_interest), len(df_interest)):
            row = df_interest.iloc[i]
            # some of nlvr data were broken
            question = row["question"]  # str
            answer = row["caption"]  # str
            text += f"##問: {question} ##答: {answer}。"

        # remove final space
        text = text[: len(text) - 1]

        # imageのロード
        img = Image.open(img_path).convert("RGB")
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
