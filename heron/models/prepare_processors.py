from typing import Dict

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlamaTokenizer,
)


def get_tokenizer(language_model_name: str) -> "Tokenizer":
    if "stabilityai/japanese-stablelm" in language_model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            padding_side="right",
            additional_special_tokens=["▁▁"],
        )
        return tokenizer

    elif "matsuo-lab/weblab" in language_model_name:
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

    elif "cyberagent/open-calm" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=True
        )
        return tokenizer

    elif "mosaicml/mpt" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    elif "meta-llama/Llama-2" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side="right", use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    elif "facebook/opt" in language_model_name:
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

    if "git" in model_type:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(
            model_config["vision_model_name"]
        )

    elif model_type == "video_blip":
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    else:
        raise NotImplementedError(f"Processor for model_type: {model_type} is not implemented.")

    processor.tokenizer = get_tokenizer(language_model_name)

    return processor
