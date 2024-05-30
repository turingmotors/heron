# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Llava Llama model configuration"""

import warnings
from typing import Union

import torch
from transformers import AutoConfig, CLIPVisionConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig


class LlavaLlamaConfig(PretrainedConfig):
    model_type = "llava_llama"
    is_composition = False

    def __init__(
        self,
        language_model_name: Union[str, None] = None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32002,
        **kwargs,
    ):
        self.language_model_name = language_model_name
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect",
                FutureWarning,
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if hasattr(self, "vision_model_name"):
            self.set_extra_configs(**kwargs)
        else:
            self.vision_config = CLIPVisionConfig()
            self.num_image_with_embedding = None

        self.text_config = LlamaConfig(language_model_name)
        self.text_config.vocab_size = vocab_size
        self.vocab_size = self.text_config.vocab_size
        self._vocab_size = self.text_config.vocab_size

        super().__init__(**kwargs)

    def set_extra_configs(
        self,
        num_image_with_embedding: int = 1,
        vision_model_name: Union[str, None] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.num_image_with_embedding = (
            None if num_image_with_embedding == 1 else num_image_with_embedding
        )
        self.vision_model_name = vision_model_name
        if "google/siglip" in vision_model_name:
            self.vision_config = AutoConfig.from_pretrained(vision_model_name).vision_config
        elif "recruit-jp/japanese-clip-vit-b-32-roberta-base" in vision_model_name:
            # required config info is same as openai clip
            self.vision_config = AutoConfig.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).vision_config
        else:
            self.vision_config = CLIPVisionConfig.from_pretrained(vision_model_name)
        self.torch_dtype = torch_dtype

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output
