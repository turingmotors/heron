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

