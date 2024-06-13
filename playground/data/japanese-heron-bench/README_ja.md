<div align="center">

# Heron-Bench

[English](README.md) | 日本語

</div>

**Heron-Bench**は、日本語VLM (Vision-Language Models) の評価ベンチマークです。日本に関連する21枚の画像を収集しました。各画像について「Conversation」、「Detail」、「Complex」の3つのカテゴリーを設定し、それぞれのカテゴリーに対して1つもしくは2つの質問を用意しました。最終的な評価データセットは102の質問から成り立っています。さらに、各画像にはアニメ、アート、文化、食、風景、ランドマーク、交通といった7つのサブカテゴリーが割り当てられています。

# データセットのダウンロード
HuggingFaceの[turing-motors/Japanese-Heron-Bench](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench)から`images/`をダウンロードし、`playground/data/japanese-heron-bench/`以下に配置します。

# 評価

1. OpenAI APIキーの設定

```bash
export OPENAI_API_KEY=sk-...
```

2. 評価スクリプト`heron_bench.sh`を実行

```bash
./scripts/heron_bench.sh
```

# VLM APIガイド

GPT-4V、Claude 3、Gemini Vision Proなどのクローズドモデルを評価する際は、[VLM APIガイド](../llava-bench-in-the-wild/ja/README_ja.md#guide-to-vlm-apis)を参照してください。

# 利用について

収集した画像は、パブリックドメインまたは[CC BY 1.0](https://creativecommons.org/licenses/by/1.0/deed.en)または[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/deed.en)のライセンスの下で提供されています。ライセンスの詳細については、[LICENSE.md](LICENCE.md)を参照してください。

# 引用

```bibtex
@misc{inoue2024heronbench,
      title={Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese}, 
      author={Yuichi Inoue and Kento Sasaki and Yuma Ochi and Kazuki Fujii and Kotaro Tanahashi and Yu Yamaguchi},
      year={2024},
      eprint={2404.07824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
