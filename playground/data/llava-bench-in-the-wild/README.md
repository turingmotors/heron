<div align="center">

# LLaVA-Bench-In-the-Wild (Japanese)

English | [日本語](./ja/README_ja.md)

</div>

LLaVA-Bench-In-the-Wild (Japanese) is the Japanese version dataset of LLaVA-Bench-In-the-Wild. It has been translated into Japanese using DeepL.

The `llava-bench-in-the-wild/en/*.jsonl` files have been copied from Hugging Face's [liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main).

# Download Dataset
Download `images/` from Hugging Face's [liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) and place them under `playground/data/llava-bench-in-the-wild/`.

## Claude 3

1. Install the Python SDK

```
pip install anthropic
```

2. Setup your ANTHROPIC API Key

Before you can use ANTHROPIC API, you must first obain an API Key. Visit [ANTHROPIC API Reference](https://docs.anthropic.com/claude/reference/getting-started-with-the-api) to create a key.

```
export ANTHROPIC_API_KEY="<YOUR_API_KEY>"
```

3. Inference

You can see the model family in the [Model Overview](https://docs.anthropic.com/claude/docs/models-overview).

```
python ./playground/scripts/inference_claude3.py
```

# License

Released under the [Apache License 2.0](./LICENSE).
