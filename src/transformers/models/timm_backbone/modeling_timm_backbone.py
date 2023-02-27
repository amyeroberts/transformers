# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import inspect
import warnings
from typing import Optional, Union, Tuple

from ...modeling_utils import PreTrainedModel, BackboneMixin
from ...modeling_outputs import BackboneOutput
from ...utils import ExplicitEnum, requires_backends, is_torch_available
from transformers.configuration_utils import PretrainedConfig


if is_torch_available():
    from torch import Tensor


class TimmBackbone(PreTrainedModel, BackboneMixin):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """

    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = PretrainedConfig

    def __init__(self, config, **kwargs):
        requires_backends(self, "timm")
        import timm

        super().__init__(config=config)
        self.config = config

        pretrained = getattr(config, "use_pretrained_backbone", None)
        if pretrained is None:
            raise ValueError("use_pretrained_backbone is not set in the config. Please set it to True or False.")

        # For timm we set features_only to True to use the model as a backbone. This
        # is currently not possible for transformer architectures.
        features_only = getattr(config, "features_only", True)
        in_chans = getattr(config, "num_channels", 3)
        # We just take the final layer by default. This matches the default for the transformers models.
        out_indices = getattr(config, "out_indices", (-1,))

        self._backbone = timm.create_model(
            config.backbone,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_chans,
            out_indices=out_indices,
            **kwargs
        )
        self._return_layers = self._backbone.return_layers
        self._all_layers = {layer['module']: str(i) for i, layer in enumerate(self._backbone.feature_info.info)}
        super()._init_backbone(config)

    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibitliy of the class in the library.
        """
        pass

    def forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs) -> Union[BackboneOutput, Tuple[Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot output attentions for timm backbones at the moment")

        if output_hidden_states:
            # We modify the return layers to include all the stages of the backbone
            self._backbone.return_layers = self._all_layers
            hidden_states = self._backbone(pixel_values, **kwargs)
            self._backbone.return_layers = self._return_layers
            feature_maps = tuple(hidden_states[i] for i in self.out_indices)
        else:
            feature_maps = self._backbone(pixel_values, **kwargs)
            hidden_states = None

        feature_maps = tuple(feature_maps)
        hidden_states = tuple(hidden_states) if hidden_states is not None else None

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return output

        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)
