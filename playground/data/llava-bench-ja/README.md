# LLaVA-Bench (COCO) 日本語版

このプログラムは本家LLaVAのリポジトリのLLaVA-Benchを日本語に対応させたものです。
具体的には、比較対象の正解回答を日本語に翻訳し、プロンプトを日本語に対応させたものであり、それ以外は基本的に本家のプログラムと同じです。

## 実行手順

1. COCO(2014)データのダウンロード

採点に使う画像のデータをCOCOからダウンロードしてください。

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
