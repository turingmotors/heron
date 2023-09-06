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

学习时，请使用 `projects` 目录下的 yaml 配置文件.<br>
例如，[projects/opt/exp001.yml]（./projects/opt/exp001.yml）的内容如下.

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

training_config "为训练设置, "model_config "为模型设置，"dataset_config_path "为数据集设置.<br>
目前支持 LLaMA、MPT 和 OPT 作为可在 `model_type` 中指定的 LLM 模块.

要开始学习, 请执行以下命令.


```bash
./scripts/run.sh
```

学习需要 GPU；我们在 Ubuntu 20.04 和 CUDA 11.7 上对系统进行了测试.

# 如何使用

您可以从 HuggingFace Hub 下载训练好的模型：[turing-motors/heron-chat-git-TinyLlama-1.1b-v0](https://huggingface.co/turing-motors/heron-chat-git-TinyLlama-1.1b-v0)<br>
有关推理和训练方法的更多信息, 请参阅 [notebooks](./notebooks).

```python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor
from heron.models.git_llm.git_llama import GitLlamaForCausalLM

device_id = 0

# prepare a pretrained model
model = GitLlamaForCausalLM.from_pretrained('turing-motors/heron-chat-git-TinyLlama-1.1b-v0')
model.eval()
model.to(f"cuda:{device_id}")

# prepare a processor
processor = AutoProcessor.from_pretrained('turing-motors/heron-chat-git-TinyLlama-1.1b-v0')

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
