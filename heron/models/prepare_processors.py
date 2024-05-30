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

from typing import Dict

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlamaTokenizer,
)

from heron.models.video_blip import VideoBlipProcessor


def get_tokenizer(language_model_name: str) -> "Tokenizer":
    if "stablelm" in language_model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            padding_side="right",
            additional_special_tokens=["▁▁"],
        )
        return tokenizer

    elif "weblab" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=True
        )
        tokenizer.add_special_tokens(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|padding|>",
                "unk_token": "<|endoftext|>",
            }
        )
        return tokenizer

    elif "ELYZA" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(language_model_name, padding_side="right")
        return tokenizer

    elif "open-calm" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=True
        )
        return tokenizer

    elif "mpt" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    elif "Llama" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=False, add_bos_token=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        # add special tokens
        special_tokens_dict = {"additional_special_tokens": ["<image>", "<pad>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    elif "Swallow-7b-NVE" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        # add special tokens
        special_tokens_dict = {"additional_special_tokens": ["<image>", "<pad>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    elif "opt" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=False
        )
        return tokenizer

    else:
        raise NotImplementedError(
            f"Tokenizer for language_model_name: {language_model_name} is not implemented."
        )


def get_processor(model_config: Dict) -> "Processor":
    language_model_name = model_config["language_model_name"]
    model_type = model_config["model_type"]

    if "llava_llm" in model_type:
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        processor.image_processor = CLIPImageProcessor.from_pretrained(
            model_config["vision_model_name"]
        )

    elif "git" in model_type or "llava" in model_type:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(
            model_config["vision_model_name"]
        )

    elif "video_blip" in model_type:
        processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    else:
        raise NotImplementedError(f"Processor for model_type: {model_type} is not implemented.")

    processor.tokenizer = get_tokenizer(language_model_name)

    return processor
