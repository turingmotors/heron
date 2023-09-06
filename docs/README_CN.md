<div align="center">

# Heron
**视觉/视频和语言模型库**

[English](../README.md) | [日本語](./README_JP.md) | 中文
</div>


Heron是一个可无缝集成多种图像/视频和语言模型的库. 此外, 它还提供在各种数据集上训练的预训练权重.

<div align="center">
<img src="../images/heron_image.png" width="50%">
</div>

Heron允许您结合各种模块配置自己的V&L模型. 可以在配置文件中配置视觉编码器, Adopter和LLM. 用于训练的分布式学习方法和数据集也可以轻松配置.

<img src="../images/build_train_model.png" width="100%">

# 如何安装
## 1. 获取存储库
```bash
git clone https://github.com/turingmotors/heron.git
cd heron
```

## 2. 设置 Python 环境
建议使用虚拟环境安装所需软件包. 如果要全局安装软件包, 请使用 `pip install -r requirements.txt` 代替.
### 2-a. Poetry (Recommended)
对于 [pyenv](https://github.com/pyenv/pyenv) 和 [Poetry](https://python-poetry.org/), 请按照以下步骤安装必要的软件包.
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
对于 [Anaconda](https://www.anaconda.com/), 请按照以下步骤安装必要的软件包.
```bash
conda create -n heron python=3.10 -y
conda activate heron
pip install --upgrade pip  # enable PEP 660 support

pip install -r requirements.txt
pip install -e .

# for development, install pre-commit
pre-commit install
```

## 3. 预申请 Llama-2 模型
要使用 Llama-2 模型, 您需要注册您的模型.
首先，请访问 [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b) 和 [Meta](https://ai.meta.com/resources/models-and-libraries/llama- downloads/) 并申请访问 llama-2 模型.

申请通过后, 使用您的 HaggingFace 账户登录.
```bash
huggingface-cli login
```

# 学习方法

目前支持 LLaMA、MPT 和 OPT 作为 LLM 模块.

```bash
./scripts/run.sh
```

# 如何使用

您可以从 HuggingFace Hub 下载训练好的模型：[Inoichan/GIT-Llama-2-7B](https://huggingface.co/Inoichan/GIT-Llama-2-7B)<br>
有关推理和训练方法的更多信息, 请参阅 [notebooks](./notebooks).

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

# 组织信息

[Turing Inc.](https://www.turing-motors.com/)

# 许可

[Apache License 2.0](../LICENSE)

# 参考信息

- [GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text)
- [Llava](https://github.com/haotian-liu/LLaVA)
- [GIT-LLM](https://github.com/Ino-Ichan/GIT-LLM)
