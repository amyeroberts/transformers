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
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np

from parameterized import parameterized
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import MobileViTFeatureExtractor


class MobileViTFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=20,
        do_center_crop=True,
        crop_size=18,
        do_flip_channel_order=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_flip_channel_order = do_flip_channel_order

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_flip_channel_order": self.do_flip_channel_order,
        }


@require_torch
@require_vision
class MobileViTFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = MobileViTFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = MobileViTFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "do_center_crop"))
        self.assertTrue(hasattr(feature_extractor, "center_crop"))
        self.assertTrue(hasattr(feature_extractor, "do_flip_channel_order"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    @parameterized.expand(
        [
            ("do_resize_True_do_center_crop_True_do_flip_channel_order_True", True, True, True),
            ("do_resize_True_do_center_crop_True_do_flip_channel_order_False", True, True, False),
            ("do_resize_True_do_center_crop_False_do_flip_channel_order_True", True, False, True),
            ("do_resize_True_do_center_crop_False_do_flip_channel_order_False", True, False, False),
            ("do_resize_False_do_center_crop_True_do_flip_channel_order_True", False, True, True),
            ("do_resize_False_do_center_crop_True_do_flip_channel_order_False", False, True, False),
            ("do_resize_False_do_center_crop_False_do_flip_channel_order_True", False, False, True),
            ("do_resize_False_do_center_crop_False_do_flip_channel_order_False", False, False, False),
        ]
    )
    def test_call_flags(self, _, do_resize, do_center_crop, do_flip_channel_order):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        feature_extractor.do_center_crop = do_center_crop
        feature_extractor.do_resize = do_resize
        feature_extractor.do_flip_channel_order = do_flip_channel_order
        # create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)

        expected_shapes = [(3, *x.size[::-1]) for x in image_inputs]
        if do_resize:
            # Same size logic inside resized
            resized_shapes = []
            for shape in expected_shapes:
                c, h, w = shape
                short, long = (w, h) if w <= h else (h, w)
                min_size = self.feature_extract_tester.size
                if short == min_size:
                    resized_shapes.append((c, h, w))
                else:
                    short, long = min_size, int(long * min_size / short)
                    resized_shape = (c, long, short) if w <= h else (c, short, long)
                    resized_shapes.append(resized_shape)
            expected_shapes = resized_shapes
        if do_center_crop:
            expected_shapes = [
                (
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.crop_size,
                    self.feature_extract_tester.crop_size,
                )
                for _ in range(self.feature_extract_tester.batch_size)
            ]

        pixel_values = feature_extractor(image_inputs, return_tensors=None)["pixel_values"]
        self.assertEqual(len(pixel_values), self.feature_extract_tester.batch_size)
        for idx, image in enumerate(pixel_values):
            self.assertEqual(image.shape, expected_shapes[idx])
            self.assertIsInstance(image, np.ndarray)
