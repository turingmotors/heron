import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..models.prepare_processors import HFProcessor


class JapaneseCSVDataset(Dataset):
    """Dataset for Custom Japanese CSV V&L Dataset learning"""

    def __init__(
        self,
        loaded_dataset: pd.DataFrame,
        processor: HFProcessor,
        max_length: int = 128,
        is_inference: bool = False,
    ):
        super(JapaneseCSVDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.unique_img_path = loaded_dataset.img_path.unique()

        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference

    @classmethod
    def create(
        cls,
        dataset_config: dict,
        processor: HFProcessor,
        split: str = "train",
        is_inference: bool = False,
    ):
        target_dataset_list = []
        if "coco" in dataset_config["dataset_names"]:
            if split == "train":
                df_train = pd.read_csv("./data/coco/df_train.csv")
                target_dataset_list.append(df_train)
            else:
                df_val = pd.read_csv("./data/coco/df_val.csv")
                target_dataset_list.append(df_val)
        elif "visual_genome" in dataset_config["dataset_names"]:
            df_vg = pd.read_csv("./data/visual_genome_ja/df_vg.csv")
            target_dataset_list.append(df_vg)
        else:
            raise ValueError(
                f"dataset_type: {dataset_config.get('dataset_type')} is not supported."
            )

        target_dataframe = pd.concat(target_dataset_list, axis=0, ignore_index=True)

        return cls(
            target_dataframe, processor, dataset_config["max_length"], is_inference=is_inference
        )

    def __len__(self) -> int:
        return len(self.unique_img_path)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        img_path = self.unique_img_path[index]

        df_interest = self.loaded_dataset[self.loaded_dataset.img_path == img_path].reset_index(
            drop=True
        )
        text = ""

        # concatenate text data
        for i in np.random.randint(0, len(df_interest), len(df_interest)):
            row = df_interest.iloc[i]
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
        text += f"##問: {question} ##答: "

        # imageのロード
        img = Image.open(img_path).convert("RGB")
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
