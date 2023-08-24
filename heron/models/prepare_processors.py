from typing import Any

from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlamaTokenizer

HFProcessor = Any


def get_processor(model_name: str, vision_model_name: str = "") -> HFProcessor:
    if "japanese-stablelm" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            padding_side="right",
            additional_special_tokens=["▁▁"],
        )
    elif "matsuo-lab/weblab" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True
        )

        processor.tokenizer.add_special_tokens(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|padding|>",
                "unk_token": "<|endoftext|>",
            }
        )
    elif "cyberagent/open-calm-7b" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True
        )
    elif "mpt" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif "llama" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=False
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif "facebook/opt" in model_name:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=False
        )
    else:
        raise NotImplementedError(f"Processor for model_name: {model_name} is not implemented.")
    return processor
