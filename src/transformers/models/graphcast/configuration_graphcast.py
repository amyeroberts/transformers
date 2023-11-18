# coding=utf-8
# Copyright 2023 Microsoft, clefourrier and The HuggingFace Inc. team. All rights reserved.
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
""" GraphCast model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


from typing import Optional, Mapping



logger = logging.get_logger(__name__)

GRAPHCAST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "deepming/graphcast-small": "https://huggingface.co/deepming/graphcast-small/resolve/main/config.json",
}



class GraphCastConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GraphCastModel`]. It is used to instantiate an
    GraphCast model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GraphCast
    [deepming/graphcast-small](https://huggingface.co/deepming/graphcast-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:


        Example:
            ```python
            >>> from transformers import GraphCastModel, GraphCastConfig

            >>> # Initializing a GraphCast graphcast-base-pcqm4mv2 style configuration
            >>> configuration = GraphCastConfig()

            >>> # Initializing a model from the deepming/graphcast-small style configuration
            >>> model = GraphCastModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```
    """

    model_type = "graphcast"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # temp
        hidden_size: int = 512,
        layer_norm_eps: float = 1e-12,
        # possible
        node_latent_dim: int = 512,
        edge_latent_dim: int = 512,
        mlp_hidden_dim: int = 512,
        mlp_num_hidden_layers: int = 3,
        num_message_passing_steps: int = 1,
        num_processor_repetitions: int = 1,
        embed_nodes: bool = True,
        embed_edges: bool = True,
        node_output_size: Optional[Mapping[str, int]] = None,
        edge_output_size: Optional[Mapping[str, int]] = None,
        include_sent_messages_in_node_update: bool = False,
        use_layer_norm: bool = True,
        activation: str = "relu",
        f32_aggregation: bool = False,
        aggregate_edges_for_nodes_fn: str = "segment_sum",
        aggregate_normalization: Optional[float] = None,
        name: str = "DeepTypedGraphNet",
        resolution: int = 0,
        mesh_size: int = 4,
        latent_size: int = 4,
        gnn_msg_steps: int = 1,
        # num_classes: int = 1,
        # num_atoms: int = 512 * 9,
        # num_edges: int = 512 * 3,
        # num_in_degree: int = 512,
        # num_out_degree: int = 512,
        # num_spatial: int = 512,
        # num_edge_dis: int = 128,
        # multi_hop_max_dist: int = 5,  # sometimes is 20
        # spatial_pos_max: int = 1024,
        # edge_type: str = "multi_hop",
        # max_nodes: int = 512,
        # share_input_output_embed: bool = False,
        # num_hidden_layers: int = 12,
        # embedding_dim: int = 768,
        # ffn_embedding_dim: int = 768,
        # num_attention_heads: int = 32,
        # dropout: float = 0.1,
        # attention_dropout: float = 0.1,
        # activation_dropout: float = 0.1,
        # layerdrop: float = 0.0,
        # encoder_normalize_before: bool = False,
        # pre_layernorm: bool = False,
        # apply_graphcast_init: bool = False,
        # activation_fn: str = "gelu",
        # embed_scale: float = None,
        # freeze_embeddings: bool = False,
        # num_trans_layers_to_freeze: int = 0,
        # traceable: bool = False,
        # q_noise: float = 0.0,
        # qn_block_size: int = 8,
        # kdim: int = None,
        # vdim: int = None,
        # bias: bool = True,
        # self_attention: bool = True,
        # pad_token_id=0,
        # bos_token_id=1,
        # eos_token_id=2,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        # self.node_latent_dim = node_latent_dim
        # self.edge_latent_dim = edge_latent_dim
        # self.mlp_hidden_dim = mlp_hidden_dim
        # self.mlp_num_hidden_layers = mlp_num_hidden_layers
        # self.num_message_passing_steps = num_message_passing_steps
        # self.num_processor_repetitions = num_processor_repetitions
        # self.embed_nodes = embed_nodes
        # self.embed_edges = embed_edges
        # self.node_output_size = node_output_size
        # self.edge_output_size = edge_output_size
        # self.include_sent_messages_in_node_update = include_sent_messages_in_node_update
        # self.use_layer_norm = use_layer_norm
        # self.activation = activation
        # self.f32_aggregation = f32_aggregation
        # self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        # self.aggregate_normalization = aggregate_normalization
        # self.name = name
        # self.resolution = resolution
        # self.mesh_size = mesh_size
        # self.latent_size = latent_size
        # self.gnn_msg_steps = gnn_msg_steps
