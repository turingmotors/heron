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

import json
import os

import fire
import numpy as np
import pandas as pd
import tqdm


def main(path_to_coco: str) -> None:
    # Set path to STAIR data
    # Images are download from here: https://cocodataset.org/#download
    # Text data are download from here: https://github.com/STAIR-Lab-CIT/STAIR-captions
    # Download data at data/coco

    path_to_coco_abs = os.path.abspath(path_to_coco) + "/"
    print(f"Path to COCO: {path_to_coco_abs}")

    # Add some pseudo question's text
    random_question_list = [
        "画像の内容を教えてください。",
        "この画像を説明できますか？",
        "画像に何が写っていますか？",
        "画像の詳細を話してください。",
        "画像に関する情報を共有して。",
        "画像を解説してもらえますか？",
        "この画像の主題は何ですか？",
        "画像を簡潔に説明してください。",
        "画像についての概要を教えて。",
        "この画像に関する基本情報を話してください。",
        "これは何の写真ですか？",
        "写真には何が写っていますか？",
        "写真について説明してください。",
        "この写真はどういう状況ですか？説明してください。",
    ]

    for target in ["train", "val"]:
        img_path_list = []
        caption_list = []
        question_list = []

        PATH_TO_JSON = path_to_coco_abs + f"stair_captions_v1.2_{target}.json"

        # load json file
        with open(PATH_TO_JSON, "r") as f:
            coco_json = json.load(f)

        # create dataframe
        # annotations to pandas DataFrame
        df_coco = pd.DataFrame(coco_json["annotations"])

        for i in tqdm.tqdm(range(len(df_coco))):
            row = df_coco.iloc[i]
            image_id = row.image_id
            img_path = path_to_coco_abs + f"{target}2014/COCO_{target}2014_{image_id:012}.jpg"
            if os.path.exists(img_path):
                img_path_list.append(img_path)
                caption_list.append(row.caption)
                q_index = np.random.randint(len(random_question_list))
                question_list.append(random_question_list[q_index])
            else:
                print(f"Fail path: {img_path}")

        df = pd.DataFrame(
            {
                "img_path": img_path_list,
                "caption": caption_list,
                "question": question_list,
            }
        )

        df.to_csv(path_to_coco_abs + f"df_{target}.csv", index=False)
        print(f"Successfully saved {target} csv file.")


if __name__ == "__main__":
    print("Start make STAIR csvs.")
    fire.Fire(main)
