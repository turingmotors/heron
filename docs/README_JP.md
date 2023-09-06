<div align="center">

# Heron
**マルチモーダルモデル学習ライブラリ**

[English](../README.md) | 日本語 | [中文](./README_CN.md)
</div>


Heronは、複数の画像/動画モデルと言語モデルをシームレスに統合するライブラリです。日本語のV&Lモデルをサポートしており、さらに様々なデータセットで学習された事前学習済みウェイトも提供します。

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
まず、[HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b)と[Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)のサイトから、llama-2モデルへのアクセスをリクエストしてください。

リクエストが承認されたら、HaggingFaceのアカウントでサインインしてください。
```bash
huggingface-cli login
```

# 学習方法

LLMモジュールとしては現在、LLaMA, MPT, OPTがサポートされています。

```bash
./scripts/run.sh
```

# 利用方法

HuggingFace Hubから学習済みモデルをダウンロードすることができます: [Inoichan/GIT-Llama-2-7B](https://huggingface.co/Inoichan/GIT-Llama-2-7B)<br>
推論・学習の方法については[notebooks](./notebooks)も参考にしてください。

```python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor
from heron.models.git_llm.git_llama import GitLlamaForCausalLM

device_id = 0

# prepare a pretrained model
model = GitLlamaForCausalLM.from_pretrained('Inoichan/GIT-Llama-2-7B')
model.eval()
model.to(f"cuda:{device_id}")

# prepare a processor
processor = AutoProcessor.from_pretrained('Inoichan/GIT-Llama-2-7B')

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

# 組織情報

[Turing株式会社](https://www.turing-motors.com/)

# ライセンス

[Apache License 2.0](../LICENSE) において公開されています。

# 参考情報

- [GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text): モデルの構成方法の着想はGITに基づいています。
- [Llava](https://github.com/haotian-liu/LLaVA): 本ライブラリはLlavaプロジェクトを参考にしています。
- [GIT-LLM](https://github.com/Ino-Ichan/GIT-LLM)
