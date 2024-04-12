<div align="center">

# LLaVA-Bench-In-the-Wild (Japanese)

English | [日本語](./ja/README_ja.md)

</div>

LLaVA-Bench-In-the-Wild (Japanese) is the Japanese version dataset of LLaVA-Bench-In-the-Wild. It has been translated into Japanese using DeepL.

The `llava-bench-in-the-wild/en/*.jsonl` files have been copied from Hugging Face's [liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main).

# Download Dataset
Download `images/` from Hugging Face's [liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) and place them under `playground/data/llava-bench-in-the-wild/`.

# Guide to VLM APIs
When evaluating Closed Models such as GPT-4V, Claude 3, and Gemini Vision Pro, please refer to the following:

## Gemini
For the full guide, please visit the [Gemini API: Quickstart with Python](https://ai.google.dev/tutorials/python_quickstart).

1. Install the Python SDK
```
pip install -q -U google-generativeai
```

2. Setup your Gemini API Key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in [Google AI Studio](https://aistudio.google.com/app/apikey).


```
export GEMINI_API_KEY=<YOUR_API_KEY>
```

3. Inference

```
python ./playground/scripts/inference_gemini.py
```

# License

Released under the [Apache License 2.0](./LICENSE).
