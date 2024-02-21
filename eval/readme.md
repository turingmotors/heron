# LLaVA-Bench (COCO) 日本語版

このプログラムは本家LLaVAのリポジトリのLLaVA-Benchを日本語に対応させたものです。
具体的には、比較対象の正解回答を日本語に翻訳し、プロンプトを日本語に対応させたものであり、それ以外は基本的に本家のプログラムと同じです。

## 実行手順

1. COCO(2014)データのダウンロード

採点に使う画像のデータをCOCOからダウンロードしてください。

```
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

2. 回答文の推論

`inference_coco_bench.ipynb`ノートブックを使って、評価対象の画像に対して回答文を生成する推論を行なってください。

3. 評価プログラムの実行

以下のコマンドによって採点スクリプトを実行してください。answer.jsonlはステップ1で出力した回答文です。score.jsonは採点プログラムによって出力されるスコアファイルです。

```
OPENAI_API_KEY="sk-..." python gpt_review.py --question qa90_questions_ja.jsonl --contex caps_boxes_coco2014_val_80.jsonl --answer-list qa90_gpt4_answer_ja_v2.jsonl sample_answer.jsonl --rule rule.json --output sample_review.json
```

4. スコアの計算と可視化

visualize.ipynbを用いて3の結果からLLaVA-Benchのスコアを算出したり、結果を比較して可視化することが可能です。

