<div align="center">

# Heron
**マルチモーダルモデル学習ライブラリ**

[English](../README.md) | 日本語 | [中文](./README_CN.md)
</div>


Heronは、複数の画像/動画モデルと言語モデルをシームレスに統合するライブラリです。日本語のV&Lモデルをサポートしており、さらに様々なデータセットで学習された事前学習済みウェイトも提供します。

異なるLLMで構築されたマルチモーダルのデモページはこちらをご覧ください。（ともに日本語対応）
- [BLIP + Japanese StableLM Base Alpha](https://huggingface.co/spaces/turing-motors/heron_chat_blip)
- [GIT + ELYZA-japanese-Llama-2](https://huggingface.co/spaces/turing-motors/heron_chat_git)

<div align="center">
<img src="../images/heron_image.png" width="50%">
</div>

Heronでは、様々なモジュールを組み合わせた独自のV&Lモデルを構成することができます。Vision Encoder、Adopter、LLMを設定ファイルで設定できます。分散学習方法やトレーニングに使用するデータセットも簡単に設定できます。

<img src="../images/build_train_model.png" width="100%">

# インストール方法
## 1. リポジトリの取得
```bash
git clone https://github.com/turingmotors/heron.git
cd heron
```

## 2. Python環境のセットアップ
必要なパッケージのインストールには仮想環境を使用することを推奨します。グローバルにパッケージをインストールしたい場合は、代わりに `pip install -r requirements.txt` を使ってください。
### 2-a. Poetry (Recommended)
[pyenv](https://github.com/pyenv/pyenv)と[Poetry](https://python-poetry.org/)の場合、次の手順で必要なパッケージをインストールしてください。
```bash
# install pyenv environment
pyenv install 3.10
pyenv local 3.10

# install packages from pyproject.toml
poetry install

# install local package
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# for development, install pre-commit
pre-commit install
``````

### 2-b. Anaconda
[Anaconda](https://www.anaconda.com/)の場合、次の手順で必要なパッケージをインストールしてください。
```bash
conda create -n heron python=3.10 -y
conda activate heron
pip install --upgrade pip  # enable PEP 660 support

pip install -r requirements.txt
pip install -e .

# for development, install pre-commit
pre-commit install
```

## 3. Llama-2モデルの事前申請
Llama-2モデルを使用するには、アクセスの申請が必要です。
まず、[Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b)と[Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)のサイトから、llama-2モデルへのアクセスをリクエストしてください。

リクエストが承認されたら、HaggingFaceのアカウントでサインインしてください。
```bash
huggingface-cli login
```

# 学習方法

学習を行う場合、`projects`ディレクトリ配下のyaml設定ファイルを使用します。<br>
例えば、[projects/opt/exp001.yml](../projects/opt/exp001.yml)の内容は次のようになっています。

```yaml
training_config:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  dataloader_num_workers: 16
  fp16: true
  optim: "adamw_torch"
  learning_rate: 5.0e-5
  logging_steps: 100
  evaluation_strategy: "steps"
  save_strategy: "steps"
  eval_steps: 4000
  save_steps: 4000
  save_total_limit: 1
  deepspeed: ./configs/deepspeed/ds_config_zero1.json
  output_dir: ./output/
  report_to: "wandb"

model_config:
  fp16: true
  pretrained_path: # None or path to model weight
  model_type: git_llm
  language_model_name: facebook/opt-350m
  vision_model_name: openai/clip-vit-base-patch16
  num_image_with_embedding: 1 # if 1, no img_temporal_embedding
  max_length: 512
  keys_to_finetune:
    - visual_projection
    - num_image_with_embedding
  keys_to_freeze: []

  use_lora: true
  lora:
    r: 8
    lora_alpha: 32
    target_modules:
      - q_proj
      - k_proj
      - v_proj
    lora_dropout: 0.01
    bias: none
    task_type: CAUSAL_LM

dataset_config_path:
  - ./configs/datasets/m3it.yaml
```

`training_config`では学習に関する設定を、`model_config`ではモデルに関する設定を、`dataset_config_path`ではデータセットに関する設定をそれぞれ行います。<br>
`model_type`に指定できるLLMモジュールとしては現在下記のものがサポートされています。今後も対応するモジュールを増やしていく予定です。

- [LLama-2](https://ai.meta.com/llama/)
- [MPT](https://github.com/mosaicml/llm-foundry)
- [OPT](https://huggingface.co/docs/transformers/model_doc/opt)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [Japanese StableLM](https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b)
- [ELYZA-japanese-Llama-2](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast)

学習を開始する場合は、次のコマンドを実行してください。

```bash
./scripts/run.sh
```

学習にはGPUが必要です。Ubuntu20.04, CUDA11.7で動作確認をしています。

# 学習方法 (Trainerなし)
Hugging Faceの `Trainer` クラスに依存しない訓練スクリプト `train_ds.py` を提供しています。<br>
例えば、[projects/opt/exp002_ds.yml](../projects/opt/exp_002_ds.yml)の内容は次のようになっています。

```yaml
training_config:
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 5
  dataloader_num_workers: 16
  learning_rate: 5.0e-5
  output_dir: ./output/
  report_to: "wandb"
  zero_stage: 2
  precision: "fp16"
  enable_tensorboard: False
  seed: 0
  weight_decay: 0.
  learning_rate_pretraining_components: 0.
  num_warmup_steps: 0.
  optim_betas:
    - 0.9
    - 0.95
  lr_scheduler_type: "cosine"
  gradient_checkpointing: False
  cpu_offload: False


model_config:
  pretrained_path: # None or path to model weight
  model_type: git_llm
  language_model_name: facebook/opt-125m
  vision_model_name: openai/clip-vit-base-patch16
  num_image_with_embedding: 1 # if 1, no img_temporal_embedding
  max_length: 512
  keys_to_finetune:
    - visual_projection
    - num_image_with_embedding
  keys_to_freeze: []

  # TODO: support LoRA
  # use_lora: false
  # lora:
  #   r: 8
  #   lora_alpha: 32
  #   target_modules:
  #     - q_proj
  #     - k_proj
  #     - v_proj
  #   lora_dropout: 0.01
  #   bias: none
  #   task_type: CAUSAL_LM

dataset_config_path:
  - ./configs/datasets/m3it_coco.yaml  # only coco dataset
```

学習を開始する場合は、次のコマンドを実行してください。

```bash
./scripts/run_ds.sh
```

# 利用方法

Hugging Face Hubから学習済みモデルをダウンロードすることができます: [turing-motors/heron-chat-git-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v0)<br>
推論・学習の方法については[notebooks](./notebooks)も参考にしてください。

```python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor
from heron.models.git_llm.git_llama import GitLlamaForCausalLM

device_id = 0

# prepare a pretrained model
model = GitLlamaForCausalLM.from_pretrained('turing-motors/heron-chat-git-ja-stablelm-base-7b-v0')
model.eval()
model.to(f"cuda:{device_id}")

# prepare a processor
processor = AutoProcessor.from_pretrained('turing-motors/heron-chat-git-ja-stablelm-base-7b-v0')

# prepare inputs
url = "https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = f"##Instruction: Please answer the following question concretely. ##Question: What is unusual about this image? Explain precisely and concretely what he is doing? ##Answer: "

# do preprocessing
inputs = processor(
    text,
    image,
    return_tensors="pt",
    truncation=True,
)
inputs = {k: v.to(f"cuda:{device_id}") for k, v in inputs.items()}

# set eos token
eos_token_id_list = [
    processor.tokenizer.pad_token_id,
    processor.tokenizer.eos_token_id,
]

# do inference
with torch.no_grad():
    out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., eos_token_id=eos_token_id_list)

# print result
print(processor.tokenizer.batch_decode(out))
```

もしZeRO-3で訓練されたモデルならば、推論用コードに次の変更を加えます。

```diff
- # prepare a pretrained model
- model = GitLlamaForCausalLM.from_pretrained(
-     'turing-motors/heron-chat-git-Llama-2-7b-v0', torch_dtype=torch.float16
- )
+ from heron.models.utils import load_model, load_pretrained_weight
+ import yaml
+ 
+ config_file = f"./projects/opt/exp002_ds.yml"
+ 
+ # get config
+ with open(config_file, "r") as i_:
+     config = yaml.safe_load(i_)
+ 
+ model = load_model(config["model_config"])
+ model.load_state_dict(torch.load('./output/opt/exp002_ds/epoch-1/pytorch_model.bin'), strict=True)
```

### 学習済みモデル一覧

|model|LLM module|adapter|size|
|:----:|:----|:----|:----|
|[heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|Japanese StableLM Base Alpha|BLIP|7B|
|[heron-chat-git-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v0)|Japanese StableLM Base Alpha|GIT|7B|
|[heron-chat-git-ELYZA-fast-7b-v0](https://huggingface.co/turing-motors/heron-chat-git-ELYZA-fast-7b-v0)|ELYZA|GIT|7B|
|[heron-preliminary-git-Llama-2-70b-v0](https://huggingface.co/turing-motors/heron-preliminary-git-Llama-2-70b-v0) *1|Llama-2|GIT|70B|
*1 アダプタの事前学習のみを実施したもの

### データセット
日本語に翻訳されたLLava-Instructデータセットです。<br>
[LLaVA-Instruct-150K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA)

# 組織情報

[Turing株式会社](https://www.turing-motors.com/)

# ライセンス

[Apache License 2.0](../LICENSE) において公開されています。

# 参考情報

- [GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text): モデルの構成方法の着想はGITに基づいています。
- [Llava](https://github.com/haotian-liu/LLaVA): 本ライブラリはLlavaプロジェクトを参考にしています。
- [GIT-LLM](https://github.com/Ino-Ichan/GIT-LLM)
- [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)
