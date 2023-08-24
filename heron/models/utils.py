import glob
from typing import Any, Optional

import numpy as np
import torch
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig

from .git_llm.git_gpt_neox import GitGPTNeoXConfig, GitGPTNeoXForCausalLM
from .git_llm.git_japanese_stablelm_alpha import (
    GitJapaneseStableLMAlphaConfig,
    GitJapaneseStableLMAlphaForCausalLM,
)
from .git_llm.git_llama import GitLlamaConfig, GitLlamaForCausalLM
from .git_llm.git_mpt import GitMptConfig, GitMptForCausalLM
from .git_llm.git_opt import GitOPTConfig, GitOPTForCausalLM

GitLLMForCausalLM = Any


def load_model(
    config: dict,
) -> GitLLMForCausalLM:
    """Loading a V&L model depending on configs"""

    model_name = config["settings"]["model_name"]
    vision_model_name = config["settings"]["vision_model_name"]
    num_image_with_embedding = config["settings"]["num_image_with_embedding"]

    # set dtype
    if config["training"]["fp16"]:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if "opt" in model_name:
        git_config = GitOPTConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitOPTForCausalLM.from_pretrained(
            model_name, config=git_config, torch_dtype=torch_dtype
        )
    elif "llama" in model_name:
        git_config = GitLlamaConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitLlamaForCausalLM.from_pretrained(
            model_name, config=git_config, torch_dtype=torch_dtype
        )
    elif "mpt" in model_name:
        git_config = GitMptConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitMptForCausalLM.from_pretrained(
            model_name, config=git_config, torch_dtype=torch_dtype
        )
    elif "japanese-stablelm" in model_name:
        git_config = GitJapaneseStableLMAlphaConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitJapaneseStableLMAlphaForCausalLM.from_pretrained(
            model_name, config=git_config, torch_dtype=torch_dtype
        )
    elif (
        "line-corporation/japanese-large-lm" in model_name
        or "matsuo-lab/weblab" in model_name
        or "cyberagent/open-calm-7b" in model_name
    ):
        git_config = GitGPTNeoXConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitGPTNeoXForCausalLM.from_pretrained(
            model_name, config=git_config, torch_dtype=torch_dtype
        )
    else:
        raise ValueError(f"{model_name} is not supported.")
    return model


def load_pretrained_weight(model: GitLLMForCausalLM, weight_path: str):
    weight = {}
    weight_path = glob.glob(f"{weight_path}/pytorch*.bin")
    for w in weight_path:
        weight_temp = torch.load(w, map_location="cpu")
        weight.update(weight_temp)
    model.load_state_dict(weight, strict=False)


def apply_lora_model(model: GitLLMForCausalLM, config: dict) -> GitLLMForCausalLM:
    """Apply LoRA"""
    model_name = config["settings"]["model_name"]
    peft_config = LoraConfig(**config["lora"])
    # apply lora only to LLM
    if "opt" in model_name:
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)
    elif "llama" in model_name:
        target_modules = []
        for m in peft_config.target_modules:
            target_modules += [
                f"model.layers.{i}.self_attn.{m}" for i in range(len(model.model.layers))
            ]

        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    elif "mpt" in model_name:
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    elif (
        "japanese-stablelm" in model_name
        or "line-corporation/japanese-large-lm" in model_name
        or "matsuo-lab/weblab" in model_name
        or "cyberagent/open-calm-7b" in model_name
    ):
        model = get_peft_model(model, peft_config)
        model.base_model.model.embed_out = model.embed_out
        # remove peft wrapper
        model = model.base_model.model
    else:
        raise ValueError(f"{model_name} is not supported.")
    return model


def set_trainable_params(model: GitLLMForCausalLM, keys_finetune: list) -> None:
    trainable_list = []
    untrainable_list = []
    for name, p in model.named_parameters():
        if np.any([k in name for k in keys_finetune]):
            p.requires_grad = True
            trainable_list.append(name)
        else:
            p.requires_grad = False
            untrainable_list.append(name)
