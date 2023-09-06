import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig

GitLLMForCausalLM = Any


def load_model(
    model_config: Dict,
) -> GitLLMForCausalLM:
    """Loading a V&L model depending on configs"""

    model_type = model_config["model_type"]
    language_model = model_config["language_model_name"]
    num_image_with_embedding = model_config["num_image_with_embedding"]

    # set dtype
    if model_config.get("fp16", False):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if model_type == "git_opt":
        from .git_llm.git_opt import GitOPTConfig, GitOPTForCausalLM

        git_config = GitOPTConfig.from_pretrained(language_model)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=model_config["vision_model_name"],
        )
        model = GitOPTForCausalLM.from_pretrained(
            language_model, config=git_config, torch_dtype=torch_dtype
        )

    elif model_type == "git_llama":
        from .git_llm.git_llama import GitLlamaConfig, GitLlamaForCausalLM

        git_config = GitLlamaConfig.from_pretrained(language_model)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=model_config["vision_model_name"],
        )
        model = GitLlamaForCausalLM.from_pretrained(
            language_model, config=git_config, torch_dtype=torch_dtype
        )

    elif model_type == "git_mpt":
        from .git_llm.git_mpt import GitMptConfig, GitMptForCausalLM

        git_config = GitMptConfig.from_pretrained(language_model)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=model_config["vision_model_name"],
        )
        model = GitMptForCausalLM.from_pretrained(
            language_model, config=git_config, torch_dtype=torch_dtype
        )

    elif model_type == "git_japanese_stablelm_alpha":
        from .git_llm.git_japanese_stablelm_alpha import (
            GitJapaneseStableLMAlphaConfig,
            GitJapaneseStableLMAlphaForCausalLM)

        git_config = GitJapaneseStableLMAlphaConfig.from_pretrained(language_model)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=model_config["vision_model_name"],
        )
        model = GitJapaneseStableLMAlphaForCausalLM.from_pretrained(
            language_model, config=git_config, torch_dtype=torch_dtype
        )

    elif model_type == "git_gpt_neox":
        from .git_llm.git_gpt_neox import (GitGPTNeoXConfig,
                                           GitGPTNeoXForCausalLM)

        git_config = GitGPTNeoXConfig.from_pretrained(language_model)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=model_config["vision_model_name"],
        )
        model = GitGPTNeoXForCausalLM.from_pretrained(
            language_model, config=git_config, torch_dtype=torch_dtype
        )

    elif model_type == "video_blip":
        from .video_blip import VideoBlipForConditionalGeneration

        model = VideoBlipForConditionalGeneration.create(
            language_model, num_frames=num_image_with_embedding, torch_dtype=torch_dtype
        )

    else:
        raise ValueError(f"{model_type} is not supported.")
    return model


def load_pretrained_weight(model: GitLLMForCausalLM, weight_path: str):
    weight = {}
    weight_path = glob.glob(f"{weight_path}/pytorch*.bin")
    for w in weight_path:
        weight_temp = torch.load(w, map_location="cpu")
        weight.update(weight_temp)
    model.load_state_dict(weight, strict=False)


def apply_lora_model(model: GitLLMForCausalLM, model_config: Dict) -> GitLLMForCausalLM:
    """Apply LoRA"""
    model_type = model_config["model_type"]
    peft_config = LoraConfig(**model_config["lora"])
    # apply lora only to LLM
    if model_type == "git_opt":
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)

    elif model_type == "git_llama":
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

    elif model_type == "git_mpt":
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model

    elif model_type == "git_gpt_neox" or model_type == "git_japanese_stablelm_alpha":
        model = get_peft_model(model, peft_config)
        model.base_model.model.embed_out = model.embed_out
        # remove peft wrapper
        model = model.base_model.model
    elif model_type == "video_blip":
        # model = get_peft_model(model, peft_config)
        model.language_model = get_peft_model(model.language_model, peft_config)
        # model.vision_model = get_peft_model(model.vision_model, peft_config)
    else:
        raise ValueError(f"{model_type} is not supported.")
    return model


def set_trainable_params(
    model: GitLLMForCausalLM,
    keys_to_finetune: List[str],
    keys_to_freeze: List[str],
    train_lora: bool = True,
) -> Tuple[List, List]:
    trainable_list = []
    untrainable_list = []

    # There is no conflict between keys_to_finetune and keys_to_freeze because one of them is empty.
    if len(keys_to_freeze) > 0 and len(keys_to_finetune) == 0:
        for name, p in model.named_parameters():
            if train_lora and "lora" in name:
                p.requires_grad = True
                trainable_list.append(name)
            elif np.any([k in name for k in keys_to_freeze]):
                p.requires_grad = False
                untrainable_list.append(name)
            else:
                p.requires_grad = True
                trainable_list.append(name)

    elif len(keys_to_finetune) > 0 and len(keys_to_freeze) == 0:
        for name, p in model.named_parameters():
            if train_lora and "lora" in name:
                p.requires_grad = True
                trainable_list.append(name)
            elif np.any([k in name for k in keys_to_finetune]):
                p.requires_grad = True
                trainable_list.append(name)
            else:
                p.requires_grad = False
                untrainable_list.append(name)

    else:
        raise ValueError("either keys_to_freeze or keys_to_finetune should be specified")

    return trainable_list, untrainable_list
