<div align="center">

# LLaVA-Bench-In-the-Wild (Japanese)

[English](../README.md) | 日本語

</div>

LLaVA-Bench-In-the-Wild (Japanese)は、LLaVA-Bench-In-the-Wildの日本語版データセットです。DeepLを用いて、日本語に翻訳しています。

`llava-bench-in-the-wild/en/*.jsonl`は、Hugging Faceの[liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main)からコピーしています。

# Download Dataset
Hugging Faceの[liuhaotian/llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)から`images/`をダウンロードして、`playground/data/llava-bench-in-the-wild/`以下に配置してください。
# Guide to VLM APIs
GPT-４V, Claude 3, Gemini Vision ProなどのClosed Modelの評価を行う場合は、以下を参照してください。

## Gemini
詳細なガイドは、[Gemini API: Quickstart with Python](https://ai.google.dev/tutorials/python_quickstart)をご覧ください。

1. Python SDKのインストール
```
pip install -q -U google-generativeai
```

2. Gemini APIキーの設定 

Gemini APIを使用するには、まずAPIキーを取得する必要があります。[Google AI Studio](https://aistudio.google.com/app/apikey)から作成してください。

```
export GEMINI_API_KEY=<YOUR_API_KEY>
```

3. 推論

```
python ./playground/scripts/inference_gemini.py
```

## Claude 3

1. Python SDKのインストール

```
pip install anthropic
```

2. ANTHROPIC APIキーの設定

ANTHROPIC APIを使用するには、まずAPIキーを取得する必要があります。APIキーを作成するには、[ANTHROPIC API Reference](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)にアクセスしてください。

```
export ANTHROPIC_API_KEY="<YOUR_API_KEY>"
```

3. 推論

[Model Overview](https://docs.anthropic.com/claude/docs/models-overview)からモデルファミリーを確認できます。

```
python ./playground/scripts/inference_claude3.py
```

# License

Released under the [Apache License 2.0](./LICENSE).
