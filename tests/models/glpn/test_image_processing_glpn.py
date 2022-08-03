# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
# See the License for the specific languag pe governing permissions and
# limitations under the License.

import numpy as np
import unittest
from parameterized import parameterized

from transformers.testing_utils import require_flax, require_vision, require_tf, require_torch
from transformers.utils import is_torch_available, is_tf_available, is_flax_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax.numpy as jnp

if is_vision_available():
    import PIL.Image

    from transformers import GLPNImageProcessor


class GLPNImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size_divisor=32,
        do_rescale=True,
        same_resolution_across_batch=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size_divisor = size_divisor
        self.same_resolution_across_batch = same_resolution_across_batch
        self.do_rescale = do_rescale

    def prepare_image_process_dict(self, do_resize=None, size_divisor=None, do_rescale=None):
        return {
            "do_resize": do_resize if do_resize is not None else self.do_resize,
            "size_divisor": size_divisor if size_divisor is not None else self.size_divisor,
            "do_rescale": do_rescale if do_rescale is not None else self.do_rescale,
        }

    def prepare_image_inputs(
        self,
        batch_size=None,
        num_channels=None,
        min_resolution=None,
        max_resolution=None,
        same_resolution_across_batch=None,
        size_divisor=None,
        return_tensors=None
    ):
        return prepare_image_inputs(
            batch_size=batch_size if batch_size is not None else self.batch_size,
            num_channels=num_channels if num_channels is not None else self.num_channels,
            min_resolution=min_resolution if min_resolution is not None else self.min_resolution,
            max_resolution=max_resolution if max_resolution is not None else self.max_resolution,
            same_resolution_across_batch=same_resolution_across_batch,
            size_divisor=size_divisor if size_divisor is not None else self.size_divisor,
            return_tensors=return_tensors
        )


class GLPNImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):

    image_processing_class = GLPNImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_process_tester = GLPNImageProcessingTester(self)

    @property
    def image_process_dict(self):
        return self.image_process_tester.prepare_image_process_dict()

    @property
    def image_inputs(self):
        return self.image_process_tester.prepare_image_inputs()

    def get_image_process_dict(self, *args, **kwargs):
        return self.image_process_tester.prepare_image_process_dict(*args, **kwargs)

    def get_image_inputs(self, *args, **kwargs):
        return self.image_process_tester.prepare_image_inputs(*args, **kwargs)

    def test_image_process_properties(self):
        image_processor = self.image_processing_class(**self.image_process_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "do_rescale"))
        self.assertTrue(hasattr(image_processor, "size_divisor"))
        self.assertTrue(hasattr(image_processor, "resample"))

    unittest.skip("Not implemented")
    def test_batch_feature(self):
        pass

    @require_torch
    def test_pytorch_inputs(self):
        image_processor_config = self.image_process_dict
        inputs = self.get_image_inputs(same_resolution_across_batch=True, return_tensors="pt")
        image_processor = self.image_processing_class(**image_processor_config)
        outputs = image_processor(inputs, return_tensors="np")

    @require_tf
    def test_tf_inputs(self):
        batch_size = 3
        num_channels = 3
        height = width = 384
        output_type_instance = np.ndarray

        inputs = self.get_image_inputs(
            batch_size=batch_size,
            num_channels=num_channels,
            same_resolution_across_batch=True,
            return_tensors="tf"
        )
        image_processor = self.image_processing_class(**self.image_process_dict)
        outputs = image_processor(inputs, return_tensors="np")
        pixel_values = outputs["pixel_values"]
        self.assertEqual(list(outputs.keys()), ["pixel_values"])
        self.assertEqual(pixel_values.shape, (batch_size, num_channels, height, width))
        self.assertIsInstance(pixel_values, output_type_instance)

    @require_flax
    def test_jax_inputs(self):
        image_processor_config = self.image_process_dict
        inputs = self.get_image_inputs(same_resolution_across_batch=True, return_tensors="jax")
        image_processor = self.image_processing_class(**image_processor_config)
        outputs = image_processor(inputs, return_tensors="np")

    @require_vision
    def test_pil_inputs(self):
        image_processor_config = self.image_process_dict
        inputs = self.get_image_inputs(same_resolution_across_batch=True, return_tensors="pil")
        image_processor = self.image_processing_class(**image_processor_config)
        outputs = image_processor(inputs, return_tensors="np")

    def test_numpy_inputs(self):
        image_processor_config = self.image_process_dict
        inputs = self.get_image_inputs(same_resolution_across_batch=True, return_tensors="np")
        image_processor = self.image_processing_class(**image_processor_config)
        outputs = image_processor(inputs, return_tensors="np")

    def test_default_behaviour(self):
        pass

    # FIXME: make safe imports
    @parameterized.expand(
        [
            # name, do_resize, do_normalize, do_rescale, expected_shape, input_type, output_type
            ("do_resize_do_normalize_input_pil_output_np", True, True, (2, 3, 64, 64), "pil", "np"),
            ("do_resize_do_normalize_input_pil_output_pt", True, True, (2, 3, 64, 64), "pil", "pt"),
            ("do_resize_do_normalize_input_pil_output_tf", True, True, (2, 3, 64, 64), "pil", "tf"),
            ("do_resize_do_normalize_input_pil_output_jax", True, True, (2, 3, 64, 64), "pil", "jax"),
            ("do_resize_no_normalize_input_pil_output_np", True, False, (2, 3, 64, 64), "pil", "np"),
            ("do_resize_no_normalize_input_pil_output_pt", True, False, (2, 3, 64, 64), "pil", "pt"),
            ("do_resize_no_normalize_input_pil_output_tf", True, False, (2, 3, 64, 64), "pil", "tf"),
            ("do_resize_no_normalize_input_pil_output_jax", True, False, (2, 3, 64, 64), "pil", "jax"),
            ("no_resize_do_normalize_input_pil_output_np", False, True, (2, 3, 70, 70), "pil", "np"),
            ("no_resize_do_normalize_input_pil_output_pt", False, True, (2, 3, 70, 70), "pil", "pt"),
            ("no_resize_do_normalize_input_pil_output_tf", False, True, (2, 3, 70, 70), "pil", "tf"),
            ("no_resize_do_normalize_input_pil_output_jax", False, True, (2, 3, 70, 70), "pil", "jax"),
            ("no_resize_no_normalize_input_pil_output_np", False, False, (2, 3, 70, 70), "pil", "np"),
            ("no_resize_no_normalize_input_pil_output_pt", False, False, (2, 3, 70, 70), "pil", "pt"),
            ("no_resize_no_normalize_input_pil_output_tf", False, False, (2, 3, 70, 70), "pil", "tf"),
            ("no_resize_no_normalize_input_pil_output_jax", False, False, (2, 3, 70, 70), "pil", "jax"),

            ("do_resize_do_normalize_input_np_output_np", True, True, (2, 3, 64, 64), "np", "np"),
            ("do_resize_do_normalize_input_np_output_pt", True, True, (2, 3, 64, 64), "np", "pt"),
            ("do_resize_do_normalize_input_np_output_tf", True, True, (2, 3, 64, 64), "np", "tf"),
            ("do_resize_do_normalize_input_np_output_jax", True, True, (2, 3, 64, 64), "np", "jax"),
            ("do_resize_no_normalize_input_np_output_np", True, False, (2, 3, 64, 64), "np", "np"),
            ("do_resize_no_normalize_input_np_output_pt", True, False, (2, 3, 64, 64), "np", "pt"),
            ("do_resize_no_normalize_input_np_output_tf", True, False, (2, 3, 64, 64), "np", "tf"),
            ("do_resize_no_normalize_input_np_output_jax", True, False, (2, 3, 64, 64), "np", "jax"),
            ("no_resize_do_normalize_input_np_output_np", False, True, (2, 3, 70, 70), "np", "np"),
            ("no_resize_do_normalize_input_np_output_pt", False, True, (2, 3, 70, 70), "np", "pt"),
            ("no_resize_do_normalize_input_np_output_tf", False, True, (2, 3, 70, 70), "np", "tf"),
            ("no_resize_do_normalize_input_np_output_jax", False, True, (2, 3, 70, 70), "np", "jax"),
            ("no_resize_no_normalize_input_np_output_np", False, False, (2, 3, 70, 70), "np", "np"),
            ("no_resize_no_normalize_input_np_output_pt", False, False, (2, 3, 70, 70), "np", "pt"),
            ("no_resize_no_normalize_input_np_output_tf", False, False, (2, 3, 70, 70), "np", "tf"),
            ("no_resize_no_normalize_input_np_output_jax", False, False, (2, 3, 70, 70), "np", "jax"),

            ("do_resize_do_normalize_input_pt_output_np", True, True, (2, 3, 64, 64), "pt", "np"),
            ("do_resize_do_normalize_input_pt_output_pt", True, True, (2, 3, 64, 64), "pt", "pt"),
            ("do_resize_do_normalize_input_pt_output_tf", True, True, (2, 3, 64, 64), "pt", "tf"),
            ("do_resize_do_normalize_input_pt_output_jax", True, True, (2, 3, 64, 64), "pt", "jax"),
            ("do_resize_no_normalize_input_pt_output_np", True, False, (2, 3, 64, 64), "pt", "np"),
            ("do_resize_no_normalize_input_pt_output_pt", True, False, (2, 3, 64, 64), "pt", "pt"),
            ("do_resize_no_normalize_input_pt_output_tf", True, False, (2, 3, 64, 64), "pt", "tf"),
            ("do_resize_no_normalize_input_pt_output_jax", True, False, (2, 3, 64, 64), "pt", "jax"),
            ("no_resize_do_normalize_input_pt_output_np", False, True, (2, 3, 70, 70), "pt", "np"),
            ("no_resize_do_normalize_input_pt_output_pt", False, True, (2, 3, 70, 70), "pt", "pt"),
            ("no_resize_do_normalize_input_pt_output_tf", False, True, (2, 3, 70, 70), "pt", "tf"),
            ("no_resize_do_normalize_input_pt_output_jax", False, True, (2, 3, 70, 70), "pt", "jax"),
            ("no_resize_no_normalize_input_pt_output_np", False, False, (2, 3, 70, 70), "pt", "np"),
            ("no_resize_no_normalize_input_pt_output_pt", False, False, (2, 3, 70, 70), "pt", "pt"),
            ("no_resize_no_normalize_input_pt_output_tf", False, False, (2, 3, 70, 70), "pt", "tf"),
            ("no_resize_no_normalize_input_pt_output_jax", False, False, (2, 3, 70, 70), "pt", "jax"),

            ("do_resize_do_normalize_input_tf_output_np", True, True, (2, 3, 64, 64), "tf", "np"),
            ("do_resize_do_normalize_input_tf_output_pt", True, True, (2, 3, 64, 64), "tf", "pt"),
            ("do_resize_do_normalize_input_tf_output_tf", True, True, (2, 3, 64, 64), "tf", "tf"),
            ("do_resize_do_normalize_input_tf_output_jax", True, True, (2, 3, 64, 64), "tf", "jax"),
            ("do_resize_no_normalize_input_tf_output_np", True, False, (2, 3, 64, 64), "tf", "np"),
            ("do_resize_no_normalize_input_tf_output_pt", True, False, (2, 3, 64, 64), "tf", "pt"),
            ("do_resize_no_normalize_input_tf_output_tf", True, False, (2, 3, 64, 64), "tf", "tf"),
            ("do_resize_no_normalize_input_tf_output_jax", True, False, (2, 3, 64, 64), "tf", "jax"),
            ("no_resize_do_normalize_input_tf_output_np", False, True, (2, 3, 70, 70), "tf", "np"),
            ("no_resize_do_normalize_input_tf_output_pt", False, True, (2, 3, 70, 70), "tf", "pt"),
            ("no_resize_do_normalize_input_tf_output_tf", False, True, (2, 3, 70, 70), "tf", "tf"),
            ("no_resize_do_normalize_input_tf_output_jax", False, True, (2, 3, 70, 70), "tf", "jax"),
            ("no_resize_no_normalize_input_tf_output_np", False, False, (2, 3, 70, 70), "tf", "np"),
            ("no_resize_no_normalize_input_tf_output_pt", False, False, (2, 3, 70, 70), "tf", "pt"),
            ("no_resize_no_normalize_input_tf_output_tf", False, False, (2, 3, 70, 70), "tf", "tf"),
            ("no_resize_no_normalize_input_tf_output_jax", False, False, (2, 3, 70, 70), "tf", "jax"),

            ("do_resize_do_normalize_input_jax_output_np", True, True, (2, 3, 64, 64), "jax", "np"),
            ("do_resize_do_normalize_input_jax_output_pt", True, True, (2, 3, 64, 64), "jax", "pt"),
            ("do_resize_do_normalize_input_jax_output_tf", True, True, (2, 3, 64, 64), "jax", "tf"),
            ("do_resize_do_normalize_input_jax_output_jax", True, True, (2, 3, 64, 64), "jax", "jax"),
            ("do_resize_no_normalize_input_jax_output_np", True, False, (2, 3, 64, 64), "jax", "np"),
            ("do_resize_no_normalize_input_jax_output_pt", True, False, (2, 3, 64, 64), "jax", "pt"),
            ("do_resize_no_normalize_input_jax_output_tf", True, False, (2, 3, 64, 64), "jax", "tf"),
            ("do_resize_no_normalize_input_jax_output_jax", True, False, (2, 3, 64, 64), "jax", "jax"),
            ("no_resize_do_normalize_input_jax_output_np", False, True, (2, 3, 70, 70), "jax", "np"),
            ("no_resize_do_normalize_input_jax_output_pt", False, True, (2, 3, 70, 70), "jax", "pt"),
            ("no_resize_do_normalize_input_jax_output_tf", False, True, (2, 3, 70, 70), "jax", "tf"),
            ("no_resize_do_normalize_input_jax_output_jax", False, True, (2, 3, 70, 70), "jax", "jax"),
            ("no_resize_no_normalize_input_jax_output_np", False, False, (2, 3, 70, 70), "jax", "np"),
            ("no_resize_no_normalize_input_jax_output_pt", False, False, (2, 3, 70, 70), "jax", "pt"),
            ("no_resize_no_normalize_input_jax_output_tf", False, False, (2, 3, 70, 70), "jax", "tf"),
            ("no_resize_no_normalize_input_jax_output_jax", False, False, (2, 3, 70, 70), "jax", "jax"),
        ]
    )
    def test_image_processor(
        self,
        name,
        do_resize,
        do_rescale,
        expected_shape,
        input_type,
        output_type,
    ):
        TYPE_NAME_INSTANCE_MAP = {
            "pil": PIL.Image.Image,
            "np": np.ndarray,
            "pt": torch.Tensor,
            "tf": tf.Tensor,
            "jax": jnp.ndarray,
        }
        image_processor_config = self.get_image_process_dict(
            do_resize=do_resize,
            do_rescale=do_rescale,
            size_divisor=32,
        )
        image_processor = self.image_processing_class(**image_processor_config)
        inputs = self.get_image_inputs(
            batch_size=2,
            max_resolution=70,
            same_resolution_across_batch=True,
            return_tensors=input_type,
        )
        outputs = image_processor(inputs, return_tensors=output_type)
        pixel_values = outputs["pixel_values"]
        self.assertIsInstance(pixel_values, TYPE_NAME_INSTANCE_MAP[str(output_type)])
        self.assertEqual(pixel_values.shape, expected_shape)

    def test_data_format(self):
        pass

    def test_fails_invalid_image(self):
        pass

    def test_fails_size_divisior_not_specified_with_do_resize(self):
        image_processor = self.image_processing_class(**self.image_process_dict)
