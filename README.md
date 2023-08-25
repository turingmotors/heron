# Heron - A Library for Vision/Video and Language models

<a href='https://huggingface.co/Inoichan/GIT-Llama-2-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> 

<img src="./images/heron_image.png" width="50%">

Welcome to "heron" repository. Heron is a library that seamlessly integrates multiple Vision and Language models, as well as Video and Language models. One of its standout features is its support for Japanese V&L models. Additionally, we provide pretrained weights trained on various datasets.


# Installation
1. Clone this repository
```bash
git clone https://github.com/turingmotors/heron.git
cd heron
```

2. Install Packages
```bash
conda create -n heron python=3.10 -y
conda activate heron
pip install --upgrade pip  # enable PEP 660 support

pip install -r requirements.txt
pip install -e .
```

## For Llama 2
First, you request access to the llama-2 models, in [huggingface page](https://huggingface.co/meta-llama/Llama-2-7b) and [facebook website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

Please sign-in the huggingface account
```bash
huggingface-cli login
```

# Training

Now we support LLaMA, MPT, and OPT as a LLM module.

```bash
./scripts/run.sh
```

# Evaluation

You can get the pretrained weight form HuggingFace Hub: [Inoichan/GIT-Llama-2-7B](https://huggingface.co/Inoichan/GIT-Llama-2-7B)<br>
See also [notebooks](./notebooks).

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

text = f"##Instruction: Please answer the following question concletely. ##Question: What is unusual about this image? Explain precisely and concletely what he is doing? ##Answer: "

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

# Acknoledge

- [GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text): The main idia of the model is based on original GIT.
- [Llava](https://github.com/haotian-liu/LLaVA): This project is learned a lot from the great Llava project.
- [GIT-LLM](https://github.com/Ino-Ichan/GIT-LLM)
- [video_blip](https://github.com/kotarotanahashi/video_blip)
