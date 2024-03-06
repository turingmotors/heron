# LLaVA-Bench (COCO) 日本語版

これは、[LLaVA](https://github.com/haotian-liu/LLaVA)で用いられているベンチマークの一つであるLLaVA-Bench (COCO)の日本語版です。
DeepL翻訳を用いて、QAのみを日本語に対応させています。

## LLaVA-Bench (COCO)
LLaVA-Bench (COCO)は、LLaVAの評価ベンチマークとして提案されました。LLaVA-Benchでは、COCO-Val-2014データセットから無作為に30枚の画像を選び、各画像について3種類の質問(conversation, detailed description, complex reasoning)を生成して、合計で90個の質問を行います。

例えば、`COCO_val2014_000000441147.jpg`に対する質問は以下のようになっています。カッコ内はLLaVA-Bench (COCO)で用いている日本語訳です。

- Conversation: What is the color of the two suitcases in the image? (画像に写っている2つのスーツケースの色は？)

- Detailed description: Analyze the image in a comprehensive and detailed manner. (包括的かつ詳細に画像を分析する。)

- Complex Reasoning: What potential factors could make these suitcases valuable? (これらのスーツケースを価値あるものにする可能性のある要素とは？)

<img src="../../../images/COCO_val2014_000000441147.jpg" width="30%">

## 実行手順

1. COCO(2014)データセットのダウンロード

COCO-Val-2014データセットをダウンロードしてください。

```
cd playground/data/llava-bench-ja/
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

2. OpenAI API Keyの設定

LLaVA-BenchではGPT-4を使用するため、OpenAI API Keyを環境変数に設定してください。
```
export OPENAI_API_KEY=sk-...
```

3. 評価プログラムの実行

`llava_bench.sh`を実行して、推論と評価を行ってください。（環境変数やConfigファイルは、実行環境に応じて変更してください）

Notebookで実行する場合：

推論は、`heron/eval/notebook/inference_coco_bench.ipynb`のノートブックでも行えます。推論の結果は、`gpt_review.py`スクリプトを実行することで評価できます。また、これらの結果を可視化する場合は、`visualize.ipynb`ノートブックを実行してください。

## Citation

```
@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```
