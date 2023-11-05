import re
import torch
import torch.nn as nn
from transformers.models.git.modeling_git import GitProjection

class MLPProjection(GitProjection):
    def __init__(self, config):
        super(MLPProjection, self).__init__(config)
        self.config = config

        if re.match(r'^mlp(\d+)x_gelu$', config.mlp_adapter):
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.mlp_adapter)
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.vision_config.hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            modules.append(nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps))
        elif re.match(r'^linear$', config.mlp_adapter):
            modules = [nn.Linear(config.vision_config.hidden_size, config.hidden_size)]
            modules.append(nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps))
        else:
            raise ValueError(f'Unknown mlp_adapter name: {config.mlp_adapter}')

        self.visual_projection = nn.Sequential(*modules)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(embeddings)
