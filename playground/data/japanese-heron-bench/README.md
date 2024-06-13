<div align="center">

# Japanese-Heron-Bench

English | [日本語](README_ja.md)

</div>

**Japanese-Heron-Bench** is a benchmark for evaluating Japanese VLMs (Vision-Language Models). We collected 21 images related to Japan. We then set up three categories for each image: Conversation, Detail, and Complex, and prepared one or two questions for each category. The final evaluation dataset consists of 102 questions. Furthermore, each image is assigned one of seven subcategories: anime, art, culture, food, landscape, landmark, and transportation.

# Download Dataset
Download `images/` from HuggingFace Dataset, [turing-motors/Japanese-Heron-Bench](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench) and place them under `playground/data/japanese-heron-bench/`.

# Evaluation
1. Setup your OpenAI API Key

```bash
export OPENAI_API_KEY=sk-...
```

2. run `heron_bench.sh` for evaluation.

```bash
./scripts/heron_bench.sh
```

# Guide to VLM APIs
When evaluating Closed Models such as GPT-4V, Claude 3, and Gemini Vision Pro, please refer to the [Guide to VLM APIs](../llava-bench-in-the-wild/README.md#guide-to-vlm-apis).

# Uses
We have collected images that are either in the public domain or licensed under Creative Commons Attribution 1.0 (CC BY 1.0) or Creative Commons Attribution 2.0 (CC BY 2.0). Please refer to the [LICENSE.md](LICENCE.md) file for details on the licenses.

# Citation

If you find Heron-Bench useful for your research and applications, please cite using this BibTex:

```bibtex
@misc{inoue2024heronbench,
      title={Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese}, 
      author={Yuichi Inoue and Kento Sasaki and Yuma Ochi and Kazuki Fujii and Kotaro Tanahashi and Yu Yamaguchi},
      year={2024},
      eprint={2404.07824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
