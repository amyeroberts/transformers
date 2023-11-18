# coding=utf-8
# Copyright 2023 Microsoft, clefourrier The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GraphCast model."""

import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphcast import GraphCastConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "deepming/graphcast-small"
_CONFIG_FOR_DOC = "GraphCastConfig"


GRAPHCAST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "deepming/graphcast-small",
    # See all GraphCast models at https://huggingface.co/models?filter=graphcast
]

class GraphCastMLP(nn.Module):
    def __init__(self, config, input_size=0):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, config.hidden_size) # FIXME - use_bias=True
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.linear_1(x)
        return x

class GraphCastBlock(nn.Module):
    def __init__(self, config, include_layernorm=True, input_size=0):
        super().__init__()
        if include_layernorm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GraphCastMLP(config, input_size=input_size)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class GraphCastEdge(nn.Module):
    def __init__(self, config, input_size=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GraphCastMLP(config, input_size=input_size)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class GraphCastGridNode(nn.Module):
    def __init__(self, config, input_size=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GraphCastMLP(config, input_size=input_size)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class GraphCastMeshNode(nn.Module):
    def __init__(self, config, input_size=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GraphCastMLP(config, input_size=input_size)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class GraphCastNodes(nn.Module): # FIXME - do we need this layer? Can it just go in the encoder directly?
    def __init__(self, config, input_size=0):
        super().__init__()
        self.grid_nodes = GraphCastGridNode(config, input_size=input_size)
        self.mesh_nodes = GraphCastMeshNode(config, input_size=input_size)

    def forward(self, x):
        x = self.grid_nodes(x)
        x = self.mesh_nodes(x)
        return x


class GraphCastEncoder(nn.Module):
    def __init__(self, config, include_layer_norm=True, input_size=0, include_nodes=True, include_edges=True):
        super().__init__()
        if include_edges:
          self.edges = GraphCastBlock(config, include_layernorm=include_layer_norm, input_size=input_size) # FIXME - remove grid2mesh in naming
        if include_nodes:
          self.nodes = GraphCastNodes(config, input_size=186)


class GraphCastProcessor(nn.Module):
    def __init__(self, config, num_edges=0, num_grid_nodes=0, num_mesh_nodes=0, input_size_edges=0, input_size_grid_nodes=0, input_size_mesh_nodes=0):
        super().__init__()
        if num_edges:
          self.edges = nn.Sequential(*[GraphCastEdge(config, input_size=input_size_edges) for _ in range(num_edges)])
        if num_grid_nodes:
          self.grid_nodes = nn.Sequential(*[GraphCastGridNode(config, input_size=input_size_grid_nodes) for _ in range(num_grid_nodes)])
        if num_mesh_nodes:
          self.mesh_nodes = nn.Sequential(*[GraphCastMeshNode(config, input_size=input_size_mesh_nodes) for _ in range(num_mesh_nodes)])


class GraphCastNodeDecoder(nn.Module):
    def __init__(self, config, input_size=0):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, config.hidden_size)
        self.linear_1 = nn.Linear(config.hidden_size, 83) #, bias=False) # FIXME - use_bias=True

    def forward(self, x):
        x = self.linear_0(x)
        x = self.linear_1(x)
        return x


class GraphCastGrid2Mesh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = GraphCastEncoder(config, input_size=4)
        self.processor = GraphCastProcessor(config, num_edges=1, num_grid_nodes=1, num_mesh_nodes=1, input_size_edges=1536, input_size_grid_nodes=512, input_size_mesh_nodes=1024)


class GraphCastMeshGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = GraphCastEncoder(config, include_layer_norm=True, input_size=4, include_nodes=False)
        self.processor = GraphCastProcessor(config, num_edges=16, num_grid_nodes=0, num_mesh_nodes=16, input_size_edges=1536, input_size_grid_nodes=512, input_size_mesh_nodes=1024)


class GraphCastMesh2Grid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = GraphCastEncoder(config, input_size=4, include_nodes=False)
        self.processor = GraphCastProcessor(config, num_edges=1, num_grid_nodes=1, num_mesh_nodes=1, input_size_edges=1536, input_size_grid_nodes=1024, input_size_mesh_nodes=512)
        self.decoder = GraphCastNodeDecoder(config, input_size=512)


class GraphCastPreTrainedModel(PreTrainedModel):
    config_class = GraphCastConfig
    base_model_prefix = "graphcast"

    # FIXME
    def _init_weights(self, module):
        """Initialize the weights"""
        # Initialize weights
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf
            pass


class GraphCastModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.grid2mesh_gnn = GraphCastGrid2Mesh(config)
        self.mesh_gnn = GraphCastMeshGNN(config)
        self.mesh2grid_gnn = GraphCastMesh2Grid(config)

    def forward(self, inputs):
        x = inputs
        x = self.grid2mesh(x)
        x = self.mesh_gnn(x)
        x = self.mesh_2_grid_gnn(x)
        return x
