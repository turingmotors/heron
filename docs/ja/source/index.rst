.. heron documentation master file, created by
   sphinx-quickstart on Tue Sep 12 17:08:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Heron
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


[日本語] | `[English] </en/latest/>`_ | `[中文] </zh/latest/>`_


Heronは、複数の画像/動画モデルと言語モデルをシームレスに統合するライブラリです。日本語のVision and Language (V&L)モデルをサポートしており、さらに様々なデータセットで学習された事前学習済みウェイトも提供します。

異なるLLMで構築されたマルチモーダルのデモページはこちらをご覧ください。（ともに日本語対応）

* `BLIP + Japanese StableLM Base Alpha <https://huggingface.co/spaces/turing-motors/heron_chat_blip>`_
* `GIT + ELYZA-japanese-Llama-2 <https://huggingface.co/spaces/turing-motors/heron_chat_git>`_

.. image:: ../../../images/heron_image.png
   :scale: 25%


Heronでは、様々なモジュールを組み合わせた独自のV&Lモデルを構成することができます。Vision Encoder、Adopter、LLMを設定ファイルで設定できます。分散学習方法やトレーニングに使用するデータセットも簡単に設定できます。

.. image:: ../../../images/build_train_model.png



インストール方法
-----------------------

1. リポジトリの取得
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/turingmotors/heron.git
   cd heron


1. Python環境のセットアップ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

必要なパッケージのインストールには仮想環境を使用することを推奨します。グローバルにパッケージをインストールしたい場合は、代わりに `pip install -r requirements.txt` を使ってください。


2-a. Poetry (Recommended)
""""""""""""""""""""""""""""""""""""""""

`pyenv <https://github.com/pyenv/pyenv>`_ と `Poetry <https://python-poetry.org/>`_ の場合、次の手順で必要なパッケージをインストールしてください。

.. code-block:: bash

   # install pyenv environment
   pyenv install 3.10
   pyenv local 3.10

   # install packages from pyproject.toml
   poetry install

   # install local package
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .

   # for development, install pre-commit
   pre-commit install


2-b. Anaconda
""""""""""""""""""""

`Anaconda <https://www.anaconda.com/>`_ の場合、次の手順で必要なパッケージをインストールしてください。

.. code-block:: bash

   conda create -n heron python=3.10 -y
   conda activate heron
   pip install --upgrade pip  # enable PEP 660 support

   pip install -r requirements.txt
   pip install -e .

   # for development, install pre-commit
   pre-commit install


.. attention::

   Llama-2モデルを使用するには、アクセスの申請が必要です。
   まず、 `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b>`_ と `Meta <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_ のサイトから、llama-2モデルへのアクセスをリクエストしてください。

   リクエストが承認されたら、HaggingFaceのアカウントでサインインしてください。

   .. code-block:: bash

      huggingface-cli login


学習方法
-----------------------

学習を行う場合、 `projects` ディレクトリ配下のyaml設定ファイルを使用します。
例えば、 `projects/opt/exp001.yml <https://github.com/turingmotors/heron/blob/main/projects/opt/exp001.yml>`_ の内容は次のようになっています。


.. code-block:: yaml

   training_config:
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 4
   num_train_epochs: 1
   dataloader_num_workers: 16
   fp16: true
   optim: "adamw_torch"
   learning_rate: 5.0e-5
   logging_steps: 100
   evaluation_strategy: "steps"
   save_strategy: "steps"
   eval_steps: 4000
   save_steps: 4000
   save_total_limit: 1
   deepspeed: ./configs/deepspeed/ds_config_zero1.json
   output_dir: ./output/
   report_to: "wandb"

   model_config:
   fp16: true
   pretrained_path: # None or path to model weight
   model_type: git_llm
   language_model_name: facebook/opt-350m
   vision_model_name: openai/clip-vit-base-patch16
   num_image_with_embedding: 1 # if 1, no img_temporal_embedding
   max_length: 512
   keys_to_finetune:
      - visual_projection
      - num_image_with_embedding
   keys_to_freeze: []

   use_lora: true
   lora:
      r: 8
      lora_alpha: 32
      target_modules:
         - q_proj
         - k_proj
         - v_proj
      lora_dropout: 0.01
      bias: none
      task_type: CAUSAL_LM

   dataset_config_path:
   - ./configs/datasets/m3it.yaml


`training_config` では学習に関する設定を、 `model_config` ではモデルに関する設定を、 `dataset_config_path` ではデータセットに関する設定をそれぞれ行います。
`model_type` に指定できるLLMモジュールとしては現在下記のものがサポートされています。今後も対応するモジュールを増やしていく予定です。

* `LLama-2 <https://ai.meta.com/llama/>`_
* `MPT <https://github.com/mosaicml/llm-foundry>`_
* `OPT <https://huggingface.co/docs/transformers/model_doc/opt>`_
* `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`_
* `Japanese StableLM <https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b>`_
* `ELYZA-japanese-Llama-2 <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast>`_

学習を開始する場合は、次のコマンドを実行してください。

.. code-block:: bash

   ./scripts/run.sh


学習にはGPUが必要です。Ubuntu20.04, CUDA11.7で動作確認をしています。


利用方法
---------

Hugging Face Hubから学習済みモデルをダウンロードすることができます: `turing-motors/heron-chat-git-ja-stablelm-base-7b-v0 <https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v0>`_
推論・学習の方法については `notebooks <https://github.com/turingmotors/heron/tree/main/notebooks>`_ も参考にしてください。

.. code-block:: python
   import requests
   from PIL import Image

   import torch
   from transformers import AutoProcessor
   from heron.models.git_llm.git_llama import GitLlamaForCausalLM

   device_id = 0

   # prepare a pretrained model
   model = GitLlamaForCausalLM.from_pretrained('turing-motors/heron-chat-git-ja-stablelm-base-7b-v0')
   model.eval()
   model.to(f"cuda:{device_id}")

   # prepare a processor
   processor = AutoProcessor.from_pretrained('turing-motors/heron-chat-git-ja-stablelm-base-7b-v0')

   # prepare inputs
   url = "https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg"
   image = Image.open(requests.get(url, stream=True).raw)

   text = f"##Instruction: Please answer the following question concretely. ##Question: What is unusual about this image? Explain precisely and concretely what he is doing? ##Answer: "

   # do preprocessing
   inputs = processor(
      text,
      image,
      return_tensors="pt",
      truncation=True,
   )
   inputs = {k: v.to(f"cuda:{device_id}") for k, v in inputs.items()}

   # set eos token
   eos_token_id_list = [
      processor.tokenizer.pad_token_id,
      processor.tokenizer.eos_token_id,
   ]

   # do inference
   with torch.no_grad():
      out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., eos_token_id=eos_token_id_list)

   # print result
   print(processor.tokenizer.batch_decode(out))


学習済みモデル一覧
------------------

.. list-table::

   * - model
     - LLM module
     - adapter
     - size
   * - `heron-chat-blip-ja-stablelm-base-7b-v0 <https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0>`_
     - Japanese StableLM Base Alpha
     - BLIP
     - 7B
   * - `heron-chat-git-ja-stablelm-base-7b-v0 <https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v0>`_
     - Japanese StableLM Base Alpha
     - GIT
     - 7B
   * - `heron-chat-git-ELYZA-fast-7b-v0 <https://huggingface.co/turing-motors/heron-chat-git-ELYZA-fast-7b-v0>`_
     - ELYZA
     - GIT
     - 7B
   * - `heron-preliminary-git-Llama-2-70b-v0 <https://huggingface.co/turing-motors/heron-preliminary-git-Llama-2-70b-v0>`_ *1
     - Llama-2
     - GIT
     - 70B

*1 アダプタの事前学習のみを実施したもの


データセット
------------

日本語に翻訳されたLLava-Instructデータセットです。
`LLaVA-Instruct-150K-JA <https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA>`_

組織情報
------------

`Turing株式会社 <https://www.turing-motors.com/>`_

ライセンス
------------

Apache License 2.0において公開されています。

参考情報
------------

* `GenerativeImage2Text <https://github.com/microsoft/GenerativeImage2Text>`_: モデルの構成方法の着想はGITに基づいています。
* `Llava <https://github.com/haotian-liu/LLaVA>`_ : 本ライブラリはLlavaプロジェクトを参考にしています。
* `GIT-LLM <https://github.com/Ino-Ichan/GIT-LLM>`_ 





* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
