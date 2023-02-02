# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow Wav2Vec2 model."""

import inspect
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFCausalLMOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
    TFXVectorOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    booleans_processing,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_wav2vec2 import Wav2Vec2Config


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2


# General docstring
_CONFIG_FOR_DOC = "Wav2Vec2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_TF_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_TF_CTC_EXPECTED_LOSS = 53.48

# Audio class docstring
_TF_SEQ_CLASS_CHECKPOINT = "superb/wav2vec2-base-superb-ks"
_TF_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_TF_SEQ_CLASS_EXPECTED_LOSS = 6.54

# Frame class docstring
_TF_FRAME_CLASS_CHECKPOINT = "anton-l/wav2vec2-base-superb-sd"
_TF_FRAME_EXPECTED_OUTPUT = [0, 0]

# Speaker Verification docstring
_TF_XVECTOR_CHECKPOINT = "anton-l/wav2vec2-base-superb-sv"
_TF_XVECTOR_EXPECTED_OUTPUT = 0.98

TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]

LARGE_NEGATIVE = -1e8


@dataclass
class TFWav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `tf.Tensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`tf.Tensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`tf.Tensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `tf.Tensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `tf.Tensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """

    loss: Optional[tf.Tensor] = None
    projected_states: tf.Tensor = None
    projected_quantized_states: tf.Tensor = None
    codevector_perplexity: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    contrastive_loss: Optional[tf.Tensor] = None
    diversity_loss: Optional[tf.Tensor] = None


@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`TFWav2Vec2BaseModelOutput`], with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    extract_features: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


def input_values_processing(func, config, input_values, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_values = tf.keras.Input(shape=(128,),
    dtype='float32', name="input_values")` otherwise the order of the tensors will not be guaranteed during the
    training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    signature.pop("kwargs", None)
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_values, (tuple, list)):
        for i, input in enumerate(input_values):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    elif isinstance(input_values, Mapping):
        if "inputs" in input_values:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_values`"
                " instead.",
                FutureWarning,
            )

            output["input_values"] = input_values.pop("inputs")

        if "decoder_cached_states" in input_values:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_values.pop("decoder_cached_states")

        for k, v in dict(input_values).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_values, tf.Tensor) or input_values is None:
            output[parameter_names[0]] = input_values
        else:
            raise ValueError(
                f"Data of type {type(input_values)} is not allowed only {allowed_types} is accepted for"
                f" {parameter_names[0]}."
            )

    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_values`
            output["input_values"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(booleans_processing(config=config, **boolean_dict))

    return output


def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> tf.Tensor:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    Adapted from [fairseq's
    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    tf.debugging.assert_less(
        mask_length,
        sequence_length,
        message=(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        ),
    )

    # compute number of masked spans in batch
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # make sure num masked indices <= sequence_length
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    num_masked_spans = tf.squeeze(num_masked_spans)

    # SpecAugment mask to fill
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    return spec_aug_mask


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWav2Vec2GroupNorm(tf.keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: tf.keras.initializers.Initializer = "zeros",
        gamma_initializer: tf.keras.initializers.Initializer = "ones",
        beta_regularizer: tf.keras.regularizers.Regularizer = None,
        gamma_regularizer: tf.keras.regularizers.Regularizer = None,
        beta_constraint: tf.keras.constraints.Constraint = None,
        gamma_constraint: tf.keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis "
                + str(self.axis)
                + " of input tensor should have a defined dimension but the layer received an input with shape "
                + str(input_shape)
                + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class TFWav2Vec2WeightNormConv1D(tf.keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.initialized = False
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            input_shape = input_shape.as_list()
            # Conv1D output shapes are checked at build time since TF 2.7, so we need to account for padding
            input_shape[-2] += self.explicit_padding * 2
            super().build(input_shape)

            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        if not self.initialized:
            self._init_norm()
            self.initialized = True

        self._normalize_kernel()

        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().call(padded_inputs)

        return output


def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.layer_norm = TFWav2Vec2GroupNorm(
            groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = TFWav2Vec2WeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name="conv",
        )
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


class TFWav2Vec2FeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if config.feat_extract_norm == "group":
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    def call(self, input_values):
        hidden_states = tf.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart->TFWav2Vec2
class TFWav2Vec2Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz * self.num_heads, tgt_len, src_len],
            message=(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {shape_list(attn_weights)}"
            ),
        )

        if attention_mask is not None:
            tf.debugging.assert_equal(
                shape_list(attention_mask),
                [bsz, 1, tgt_len, src_len],
                message=(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {shape_list(attention_mask)}"
                ),
            )

            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = stable_softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.num_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.num_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )

            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)

        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz * self.num_heads, tgt_len, self.head_dim],
            message=(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {shape_list(attn_output)}"
            ),
        )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value


class TFWav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)

        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states


class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [
            TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TFWav2Vec2GumbelVectorQuantizer(tf.keras.layers.Layer):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        self.config = config

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )
        self.weight_proj = tf.keras.layers.Dense(self.num_groups * self.num_vars, name="weight_proj")

        # can be decayed for training
        self.temperature = 2

    def build(self, input_shape):
        # storage for codebook variables (codewords)
        self.codevectors = self.add_weight(
            shape=(1, self.config.num_groups * self.config.num_vars, self.config.codevector_dim // self.num_groups),
            # initializer="uniform",  # FIXME
            trainable=True,
            name="codevectors",
        )
        return super().build(input_shape)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)  # FIXME
            probs = tf.where(mask_extended, probs, tf.zeros_like(probs))
            marginal_probs = tf.reduce_sum(probs, axis=0) / tf.reduce_sum(mask)
        else:
            marginal_probs = tf.reduce_mean(probs, axis=0)

        perplexity = tf.reduce_sum(
            tf.math.exp(-tf.reduce_sum(marginal_probs * tf.math.log(marginal_probs + 1e-7), axis=-1))
        )
        return perplexity

    def call(self, hidden_states, mask_time_indices=None, training=False):
        batch_size, sequence_length, hidden_size = tf.shape(hidden_states)

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = tf.reshape(hidden_states, (batch_size * sequence_length * self.num_groups, -1))

        if training:
            # sample code vector probs via gumbel in differentiateable way
            import tensorflow_probability as tfp  # FIXME

            # codevector_probs = nn.functional.gumbel_softmax(
            #     hidden_states.float(), tau=self.temperature, hard=True
            # ).type_as(hidden_states)
            gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, logits=hidden_states, validate_args=True
            )
            codevector_probs = gumbel_dist.sample()
            codevector_probs = tf.cast(codevector_probs, dtype=hidden_states.dtype)

            # compute perplexity
            codevector_soft_dist = stable_softmax(
                tf.cast(tf.reshape(hidden_states, (batch_size * sequence_length, self.num_groups, -1)), tf.float32),
                axis=-1,
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = tf.argmax(hidden_states, axis=-1)
            codevector_probs = tf.zeros_like(hidden_states)
            # codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
            #     -1, codevector_idx.view(-1, 1), 1.0
            # )  #FIXME - needs safe update as index oob won't throw error
            codevector_probs = tf.scatter_nd(codevector_idx, 1.0, codevector_probs.shape)
            codevector_probs = tf.reshape(codevector_probs, (batch_size * sequence_length, self.num_groups, -1))

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = tf.reshape(codevector_probs, (batch_size * sequence_length, -1))
        # use probs to retrieve codevectors
        codevectors_per_group = tf.expand_dims(codevector_probs, -1) * self.codevectors
        codevectors = tf.reshape(
            codevectors_per_group, (batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        )
        codevectors = tf.reduce_sum(codevectors, axis=-2)
        codevectors = tf.reshape(codevectors, (batch_size, sequence_length, -1))

        return codevectors, perplexity


@keras_serializable
class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name="feature_extractor")
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFWav2Vec2Encoder(config, name="encoder")

    def build(self, input_shape: tf.TensorShape):
        self.masked_spec_embed = self.add_weight(
            shape=(self.config.hidden_size,), initializer="uniform", trainable=True, name="masked_spec_embed"
        )

        super().build(input_shape)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: Optional[tf.Tensor] = None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        elif self.config.mask_time_prob > 0:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                min_masks=2,
            )
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        # apply SpecAugment along feature axis
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
            )
            hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)

        return hidden_states

    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs: Any,
    ):
        inputs = input_values_processing(
            func=self.call,
            config=self.config,
            input_values=input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        extract_features = self.feature_extractor(
            tf.cast(inputs["input_values"], tf.float32), training=inputs["training"]
        )
        # extract_features = tf.transpose(extract_features, perm=(0, 2, 1))

        if inputs["attention_mask"] is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(inputs["attention_mask"], -1))

            attention_mask = tf.sequence_mask(
                output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype
            )

        hidden_states, extract_features = self.feature_projection(extract_features, training=inputs["training"])

        mask_time_indices = kwargs.get("mask_time_indices", None)
        if inputs["training"]:
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        hidden_states = encoder_outputs[0]

        if not inputs["return_dict"]:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return TFWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        pad_token = 0.0
        input_values = tf.convert_to_tensor(np.random.rand(1, 16000), tf.float32)
        dummy_inputs = {
            "input_values": input_values,
            "attention_mask": tf.cast(tf.not_equal(input_values, pad_token), tf.float32),
        }
        return dummy_inputs

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tine this model, you need a GPU or a TPU"
        )

    @tf.function(
        input_signature=[
            {
                "input_values": tf.TensorSpec((None, None), tf.float32, name="input_values"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
                "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(input_values=inputs, training=False)

        return self.serving_output(output)


WAV_2_VEC_2_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_values` only and nothing else: `model(input_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_values": input_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_values` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_values` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings("""TFWav2Vec2Model with a quantizer and `VQ` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2ForPreTraining(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.wav2vec2 = TFWav2Vec2Model(config, name="wav2vec2")
        self.dropout_features = tf.keras.layers.Dropout(config.feat_quantizer_dropout, name="dropout_features")

        self.quantizer = TFWav2Vec2GumbelVectorQuantizer(config, name="quantizer")

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_hid = tf.keras.layers.Dense(config.proj_codevector_dim, name="project_hid")
        self.project_q = tf.keras.layers.Dense(config.proj_codevector_dim, name="project_q")

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    @staticmethod
    def compute_contrastive_logits(
        target_features: tf.Tensor,
        negative_features: tf.Tensor,
        predicted_features: tf.Tensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = tf.concat([target_features, negative_features], axis=0)

        logits = tf.cast(
            tf.keras.losses.cosine_similarity(
                tf.cast(predicted_features, tf.float32), tf.cast(target_features, tf.float32), axis=-1
            ),
            target_features.dtype,
        )

        # apply temperature
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFWav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_values: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor] = None,
        mask_time_indices: Optional[tf.Tensor] = None,
        sampled_negative_indices: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFWav2Vec2ForPreTrainingOutput]:
        r"""
        mask_time_indices (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`tf.Tensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoFeatureExtractor, TFWav2Vec2ForPreTraining
        >>> from transformers.models.wav2vec2.modeling_tf_wav2vec2 import (
        ...     _compute_mask_indices,
        ...     _sample_negative_indices,
        ... )
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        >>> model = TFWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="tf").input_values  # Batch size 1

        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = tf.shape(input_values)
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
        >>> mask_time_indices = _compute_mask_indices(
        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        ... )
        >>> sampled_negative_indices = _sample_negative_indices(
        ...     features_shape=(batch_size, sequence_length),
        ...     num_negatives=model.config.num_negatives,
        ...     mask_time_indices=mask_time_indices,
        ... )
        >>> mask_time_indices = tf.Tensor(data=mask_time_indices, device=input_values.device, dtype=tf.int64)
        >>> sampled_negative_indices = tf.Tensor(
        ...     data=sampled_negative_indices, device=input_values.device, dtype=tf.int64
        ... )

        >>> outputs = model(input_values, mask_time_indices=mask_time_indices)

        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = tf.keras.losses.cosine_similarity(
        ...     outputs.projected_states, outputs.projected_quantized_states, axis=-1
        ... )

        >>> # show that cosine similarity is much higher than random
        >>> tf.reduce_sum(cosine_sim[tf.cast(mask_time_indices, tf.bool)]) > 0.5
        tensor(True)

        >>> # for contrastive loss training model should be put into train mode
        >>> # model = model.train()  # FIXME
        >>> loss = model(
        ...     input_values,
        ...     mask_time_indices=mask_time_indices,
        ...     sampled_negative_indices=sampled_negative_indices,
        ...     training=True,  # FIXME
        ... ).loss
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = tf.cast(mask_time_indices, tf.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = shape_list(quantized_features)

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = tf.reshape(quantized_features, (-1, hidden_size))[
                tf.reshape(tf.cast(sampled_negative_indices, tf.int64), (-1,))
            ]
            negative_quantized_features = tf.reshape(
                negative_quantized_features, (batch_size, sequence_length, -1, hidden_size)
            )
            negative_quantized_features = tf.transpose(negative_quantized_features, (2, 0, 1, 3))

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = tf.reduce_all(quantized_features == negative_quantized_features, axis=-1)

            if tf.reduce_any(neg_is_pos):
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = tf.transpose(logits, (2, 1, 0))
            logits = tf.reshape(logits, (-1, tf.shape(logits)[0]))

            target = (1 - tf.cast(mask_time_indices, tf.int64)) * -100
            target = tf.transpose(target, (1, 0))
            target = tf.reshape(target, (-1,))

            contrastive_loss = tf.keras.losses.cross_entropy(logits.float(), target, reduction="sum")

            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * tf.reduce_sum(
                mask_time_indices
            )

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return TFWav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )


@add_start_docstrings(
    "The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, TFWav2Vec2Model
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""

        inputs = input_values_processing(
            func=self.call,
            config=self.config,
            input_values=input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        inputs["output_hidden_states"] = (
            inputs["output_hidden_states"] if inputs["output_hidden_states"] else self.config.output_hidden_states
        )
        inputs["output_attentions"] = (
            inputs["output_attentions"] if inputs["output_attentions"] else self.config.output_attentions
        )
        inputs["return_dict"] = inputs["return_dict"] if inputs["return_dict"] else self.config.return_dict

        outputs = self.wav2vec2(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs

    def serving_output(self, output):
        hidden_states = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attentions = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFWav2Vec2BaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            extract_features=output.extract_features,
            hidden_states=hidden_states,
            attentions=attentions,
        )


@add_start_docstrings(
    """TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    @unpack_inputs
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_values` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoProcessor, TFWav2Vec2ForCTC
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> logits = model(input_values).logits
        >>> predicted_ids = tf.argmax(logits, axis=-1)

        >>> transcription = processor.decode(predicted_ids[0])

        >>> # compute loss
        >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

        >>> # Pass transcription as `text` to encode labels
        >>> labels = processor(text=transcription, return_tensors="tf").input_ids

        >>> loss = model(input_values, labels=labels).loss
        ```"""
        inputs = input_values_processing(
            func=self.call,
            config=self.config,
            input_values=input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        outputs = self.wav2vec2(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        logits = self.lm_head(hidden_states)

        if labels is not None:

            if tf.reduce_max(labels) >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            attention_mask = (
                inputs["attention_mask"]
                if inputs["attention_mask"] is not None
                else tf.ones_like(inputs["input_values"], dtype=tf.float32)
            )
            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, axis=-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = tf.cast(labels >= 0, tf.int32)
            target_lengths = tf.reduce_sum(labels_mask, axis=-1)

            loss = tf.nn.ctc_loss(
                logits=logits,
                labels=labels,
                logit_length=input_lengths,
                label_length=target_lengths,
                blank_index=self.config.pad_token_id,
                logits_time_major=False,
            )

            if self.config.ctc_loss_reduction == "sum":
                loss = tf.reduce_sum(loss)
            if self.config.ctc_loss_reduction == "mean":
                loss = tf.reduce_mean(loss)

            loss = tf.reshape(loss, (1,))
        else:
            loss = None

        if not inputs["return_dict"]:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFCausalLMOutput) -> TFCausalLMOutput:
        hidden_states = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attentions = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        return TFCausalLMOutput(logits=output.logits, hidden_states=hidden_states, attentions=attentions)


@add_start_docstrings(
    """
    TFWav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.config = config
        self.wav2vec2 = TFWav2Vec2Model(config, name="wav2vec2")
        self.projector = tf.keras.layers.Dense(config.classifier_proj_size, name="projector")
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")

    def build(self, input_shape):
        if self.config.use_weighted_layer_sum:
            num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
            layer_weights = tf.ones(num_layers) / num_layers
            self.layer_weights = self.add_weight(
                shape=(num_layers,),
                initializer=tf.keras.initializers.Constant(layer_weights),
                trainable=True,  # FIXME
                name="layer_weights",
            )
        return super().build(input_shape)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # FIXME
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_TF_SEQ_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_TF_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TF_SEQ_CLASS_EXPECTED_LOSS,
    )
    @unpack_inputs
    def call(
        self,
        input_values: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = stable_softmax(self.layer_weights, axis=-1)
            hidden_states = hidden_states * tf.reshape(norm_weights, (-1, 1, 1))
            hidden_states = tf.reduce_sum(hidden_states, axis=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(tf.shape(hidden_states)[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = tf.reduce_sum(hidden_states, axis=1) / tf.reshape(
                tf.reduce_sum(padding_mask, axis=1), (-1, 1)
            )

        logits = self.classifier(pooled_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForAudioFrameClassification(TFWav2Vec2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = TFWav2Vec2Model(config, "wav2vec2")
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")
        self.num_labels = config.num_labels

    def build(self, input_shape):
        if self.config.use_weighted_layer_sum:
            num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
            layer_weights = tf.ones(num_layers) / num_layers
            self.layer_weights = self.add_weight(
                shape=(num_layers,),
                initializer=tf.keras.initializers.Constant(layer_weights),
                trainable=True,  # FIXME
                name="layer_weights",
            )
        return super().build(input_shape)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # FIXME
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_TF_FRAME_CLASS_CHECKPOINT,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_TF_FRAME_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        input_values: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = stable_softmax(self.layer_weights, axis=-1)
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, (-1, 1, 1)), axis=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), tf.argmax(labels.view(-1, self.num_labels), axis=1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TFAMSoftmaxLoss(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()  # FIXME nn.CrossEntropyLoss()

    def build(self, input_shape):
        # self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.weight = self.add_weight(
            name="weight",
            shape=(self.input_dim, self.num_labels),
            initializer="glorot_uniform",  # FIXME
            trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states, labels):
        labels = labels.flatten()
        weight = tf.linalg.normalize(self.weight, ord=2, axis=0)  # check ord, axis
        hidden_states = tf.linalg.normalize(hidden_states, ord=2, axis=1)  # check ord, axis
        cos_theta = tf.matmul(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = tf.one_hot(labels, self.num_labels)
        logits = self.scale * tf.where(tf.cast(onehot, tf.bool), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


class TFTDNNLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = tf.keras.layers.Dense(self.out_conv_dim, name="kernel")
        self.activation = tf.keras.activations.relu

    def call(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        # nn.functional.unfold - FIXME
        hidden_states = tf.image.extract_patches(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = tf.transpose(hidden_states, (0, 2, 1))
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForXVector(TFWav2Vec2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        self.projector = tf.keras.layers.Dense(config.tdnn_dim[0], name="projector")

        self.tdnn = [TFTDNNLayer(config, i, name=f"layer.{i}") for i in range(len(config.tdnn_dim))]

        self.feature_extractor = tf.keras.layers.Dense(config.xvector_output_dim, name="feature_extractor")
        self.classifier = tf.keras.layers.Dense(config.xvector_output_dim, name="classifier")
        self.objective = TFAMSoftmaxLoss(config.xvector_output_dim, config.num_labels, name="objective")

    def build(self, input_shape):
        if self.config.use_weighted_layer_sum:
            num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
            layer_weights = tf.ones(num_layers) / num_layers
            self.layer_weights = self.add_weight(
                shape=(self.config.num_hidden_layers + 1,),
                trainable=True,  # FIXME
                name="layer_weights",
            )
            self.layer_weights.assign(layer_weights)
        return super().build(input_shape)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # FIXME
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[tf.Tensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_TF_XVECTOR_CHECKPOINT,
        output_type=TFXVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_TF_XVECTOR_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        input_values: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
    ) -> Union[Tuple, TFXVectorOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = stable_softmax(self.layer_weights, axis=-1)
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, (-1, 1, 1)), axis=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = tf.reduce_mean(hidden_states, axis=1)
            std_features = tf.math.reduce_std(hidden_states, axis=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(axis=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(tf.reduce_mean(hidden_states[i, :length], axis=0))
                std_features.append(tf.math.reduce_std(hidden_states[i, :length], axis=0))
            mean_features = tf.stack(mean_features)
            std_features = tf.stack(std_features)
        statistic_pooling = tf.concat([mean_features, std_features], axis=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TFXVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
