# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_timm_available,
    is_vision_available,
)


_import_structure = {"configuration_detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig"]}

if is_vision_available():
    _import_structure["feature_extraction_detr"] = ["DetrFeatureExtractor"]

if is_timm_available():
    _import_structure["modeling_detr"] = [
        "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DetrForObjectDetection",
        "DetrForSegmentation",
        "DetrModel",
        "DetrPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_detr"] = [
        "TF_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDetrForObjectDetection",
        "TFDetrForSegmentation",
        "TFDetrModel",
        "TFDetrPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_detr import DetrFeatureExtractor

    try:
        if not is_timm_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_detr import (
            DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_detr import (
            TF_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDetrForObjectDetection,
            TFDetrForSegmentation,
            TFDetrModel,
            TFDetrPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
