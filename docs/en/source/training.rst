Training
-----------------------


For learning, use the yaml configuration file under the `projects` directory.
For example, the contents of `projects/opt/exp001.yml <https://github.com/turingmotors/heron/blob/main/projects/opt/exp001.yml>`_ has the following contents:

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



`training_config` sets the training configuration, `model_config` sets the model configuration, and `dataset_config_path` sets the dataset configuration.
The following LLM modules are currently supported for `model_type`. We plan to add more supported modules in the future.


* `LLama-2 <https://ai.meta.com/llama/>`_
* `MPT <https://github.com/mosaicml/llm-foundry>`_
* `OPT <https://huggingface.co/docs/transformers/model_doc/opt>`_
* `GPT-NeoX <https://github.com/EleutherAI/gpt-neox>`_
* `Japanese StableLM <https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b>`_
* `ELYZA-japanese-Llama-2 <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast>`_

To start learning, execute the following command.

.. code-block:: bash

   ./scripts/run.sh


GPU is required for learning; we have tested on Ubuntu 20.04, CUDA 11.7.
