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
import pandas as pd
import tqdm


def main(path_to_visual_genome: str):
    # Set path to visual genome data
    # Images are download from here: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
    # Text data are download from here: https://github.com/yahoojapan/ja-vg-vqa
    # Download data at data/visual_genome_ja

    PATH_TO_VISUAL_GENOME = os.path.abspath(path_to_visual_genome) + "/"

    with open(PATH_TO_VISUAL_GENOME + "question_answers.json", "r") as f:
        v_g = json.load(f)

    # Extract question/answer pairs
    qas_list = []
    for data in v_g:
        qas_list.extend(data["qas"])
    d_vg = pd.DataFrame(qas_list)

    img_path_list = []
    caption_list = []
    question_list = []

    for i in tqdm.tqdm(range(len(d_vg))):
        row = d_vg.iloc[i]

        image_id = row.image_id
        img_path = PATH_TO_VISUAL_GENOME + f"VG_100K/{image_id}.jpg"

        if os.path.exists(img_path):
            img_path_list.append(img_path)
            caption_list.append(row.answer)
            question_list.append(row.question)
        else:
            print(f"Fail path: {img_path}")

    df = pd.DataFrame(
        {
            "img_path": img_path_list,
            "caption": caption_list,
            "question": question_list,
        }
    )

    df.to_csv(PATH_TO_VISUAL_GENOME + "df_vg.csv", index=False)
    print("Successfully saved japanese visual genome csv file.")


if __name__ == "__main__":
    fire.Fire(main)
