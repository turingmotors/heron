如何使用
------------------

您可以从 Hugging Face Hub 下载训练好的模型： `turing-motors/heron-chat-git-ja-stablelm-base-7b-v0 <https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v0>`_ 
有关推理和训练方法的更多信息, 请参阅 `notebooks <https://github.com/turingmotors/heron/tree/main/notebooks>`_.

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
