# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the TensorFlow Beit model. """


import collections.abc
import inspect
import unittest

import numpy as np
from datasets import load_dataset
from packaging import version

from transformers import BeitConfig
from transformers.file_utils import cached_property, is_tf_available, is_vision_available
from transformers.testing_utils import require_tf, require_vision, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFBeitForImageClassification,
        TFBeitForMaskedImageModeling,
        TFBeitForSemanticSegmentation,
        TFBeitModel,
    )
    from transformers.models.beit.modeling_tf_beit import TF_BEIT_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    import PIL
    from PIL import Image

    from transformers import BeitImageProcessor


class TFBeitModelTester:
    def __init__(
        self,
        parent,
        vocab_size=100,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        out_indices=[0, 1, 2, 3],
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.out_indices = out_indices
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return BeitConfig(
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = TFBeitModel(config=config)
        result = model(pixel_values, training=False)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (
            self.image_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.image_size, self.image_size)
        )
        patch_size = (
            self.patch_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.patch_size, self.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_masked_lm(self, config, pixel_values, labels, pixel_labels):
        model = TFBeitForMaskedImageModeling(config=config)
        result = model(pixel_values, training=False)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (
            self.image_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.image_size, self.image_size)
        )
        patch_size = (
            self.patch_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.patch_size, self.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, self.vocab_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.type_sequence_label_size
        model = TFBeitForImageClassification(config)
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = TFBeitForImageClassification(config)
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = TFBeitForSemanticSegmentation(config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_keras_fit(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, _, _ = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "labels": tf.zeros((self.batch_size))}
        return config, inputs_dict


@require_tf
class TFBeitModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Beit does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (TFBeitModel, TFBeitForImageClassification, TFBeitForMaskedImageModeling, TFBeitForSemanticSegmentation)
        if is_tf_available()
        else ()
    )

    test_pruning = False
    test_onnx = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = TFBeitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BeitConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Beit does not use inputs_embeds")
    def test_inputs_embeds(self):
        # Beit does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (tf.keras.layers.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Layer))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    @unittest.skip("Test was written for TF 1.x and isn't really relevant here")
    def test_compile_tf_model(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # in Beit, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )
        patch_size = (
            self.model_tester.patch_size
            if isinstance(self.model_tester.patch_size, collections.abc.Iterable)
            else (self.model_tester.patch_size, self.model_tester.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # Beit has a different seq_length
            image_size = (
                self.model_tester.image_size
                if isinstance(self.model_tester.image_size, collections.abc.Iterable)
                else (self.model_tester.image_size, self.model_tester.image_size)
            )
            patch_size = (
                self.model_tester.patch_size
                if isinstance(self.model_tester.patch_size, collections.abc.Iterable)
                else (self.model_tester.patch_size, self.model_tester.patch_size)
            )
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # Overriding this method since the base method won't be compatible with Data2VecVision.
    def test_keras_fit(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Since `TFData2VecVisionModel` cannot operate with the default `fit()` method.
            if model_class.__name__ != "TFBeitModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    # Test that model correctly compute the loss with kwargs
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()

                    label_names = {"labels"}
                    self.assertGreater(len(label_names), 0, msg="No matching label names found!")
                    labels = {key: val for key, val in prepared_for_class.items() if key in label_names}
                    inputs_minus_labels = {
                        key: val for key, val in prepared_for_class.items() if key not in label_names
                    }
                    self.assertGreater(len(inputs_minus_labels), 0)
                    model.compile(optimizer=tf.keras.optimizers.SGD(0.0), run_eagerly=True)

                    # Make sure the model fits without crashing regardless of where we pass the labels
                    history1 = model.fit(
                        prepared_for_class,
                        validation_data=prepared_for_class,
                        steps_per_epoch=1,
                        validation_steps=1,
                        shuffle=False,
                    )
                    val_loss1 = history1.history["val_loss"][0]
                    history2 = model.fit(
                        inputs_minus_labels,
                        labels,
                        validation_data=(inputs_minus_labels, labels),
                        steps_per_epoch=1,
                        validation_steps=1,
                        shuffle=False,
                    )
                    val_loss2 = history2.history["val_loss"][0]
                    self.assertTrue(np.allclose(val_loss1, val_loss2, atol=1e-2, rtol=1e-3))

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=2e-4, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as semseg models tend to diverge a bit more
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol, name, attributes)

    # Overriding this method since the base method won't be compatible with Data2VecVision.
    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Since `TFData2VecVisionModel` won't have labels against which we
            # could compute loss.
            if model_class.__name__ != "TFBeitModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    # The number of elements in the loss should be the same as the number of elements in the label
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()
                    added_label = prepared_for_class[
                        sorted(list(prepared_for_class.keys() - inputs_dict.keys()), reverse=True)[0]
                    ]
                    loss_size = tf.size(added_label)

                    # Test that model correctly compute the loss with kwargs
                    possible_input_names = {"input_ids", "pixel_values", "input_features"}
                    input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
                    model_input = prepared_for_class.pop(input_name)

                    loss = model(model_input, **prepared_for_class)[0]
                    self.assertEqual(loss.shape, [loss_size])

                    # Test that model correctly compute the loss with a dict
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()
                    loss = model(**prepared_for_class)[0]
                    self.assertEqual(loss.shape, [loss_size])

                    # Test that model correctly compute the loss with a tuple
                    label_keys = prepared_for_class.keys() - inputs_dict.keys()
                    signature = inspect.signature(model.call).parameters
                    signature_names = list(signature.keys())

                    # Create a dictionary holding the location of the tensors in the tuple
                    tuple_index_mapping = {0: input_name}
                    for label_key in label_keys:
                        label_key_index = signature_names.index(label_key)
                        tuple_index_mapping[label_key_index] = label_key
                    sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
                    # Initialize a list with their default values, update the values and convert to a tuple
                    list_input = []

                    for name in signature_names:
                        if name != "kwargs":
                            list_input.append(signature[name].default)

                    for index, value in sorted_tuple_index_mapping:
                        list_input[index] = prepared_for_class[value]

                    tuple_input = tuple(list_input)

                    # Send to model
                    loss = model(tuple_input[:-1])[0]

                    self.assertEqual(loss.shape, [loss_size])

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_BEIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBeitModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class TFBeitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k", from_pt=True)
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_masked_image_modeling_head(self):
        model = TFBeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k", from_pt=True)

        image_processor = self.default_image_processor
        image = prepare_img()
        pixel_values = image_processor(images=image, return_tensors="tf")

        # prepare bool_masked_pos
        bool_masked_pos = tf.ones((1, 196), dtype=tf.dtypes.bool)

        # forward pass
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.convert_to_tensor([1, 196, 8192])
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = tf.convert_to_tensor(
            [[-3.2437, 0.5072, -13.9174], [-3.2456, 0.4948, -13.9401], [-3.2033, 0.5121, -13.8550]]
        )

        tf.debugging.assert_near(logits[:3, :3], expected_slice, atol=1e-4)

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = TFBeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", from_pt=True)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.convert_to_tensor([1, 1000])
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = tf.convert_to_tensor([-1.2385, -1.0987, -1.0108])

        tf.debugging.assert_near(logits[0, :3], expected_slice, atol=1e-4)

        expected_class_idx = 281
        self.assertEqual(tf.math.argmax(logits, -1), expected_class_idx)

    @slow
    def test_inference_image_classification_head_imagenet_22k(self):
        model = TFBeitForImageClassification.from_pretrained(
            "microsoft/beit-large-patch16-224-pt22k-ft22k", from_pt=True
        )

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.convert_to_tensor([1, 21841])
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = tf.convert_to_tensor([1.6881, -0.2787, 0.5901])

        tf.debugging.assert_near(logits[0, :3], expected_slice, atol=1e-4)

        expected_class_idx = 2396
        self.assertEqual(tf.math.argmax(logits, -1), expected_class_idx)

    @slow
    def test_inference_semantic_segmentation(self):
        model = TFBeitForSemanticSegmentation.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640", from_pt=True
        )

        image_processor = BeitImageProcessor(do_resize=True, size=640, do_center_crop=False)

        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.convert_to_tensor([1, 150, 160, 160])
        self.assertEqual(logits.shape, expected_shape)

        is_pillow_less_than_9 = version.parse(PIL.__version__) < version.parse("9.0.0")

        if is_pillow_less_than_9:
            expected_slice = tf.convert_to_tensor(
                [
                    [[-4.9225, -2.3954, -3.0522], [-2.8822, -1.0046, -1.7561], [-2.9549, -1.3228, -2.1347]],
                    [[-5.8168, -3.4129, -4.0778], [-3.8651, -2.2214, -3.0277], [-3.8356, -2.4643, -3.3535]],
                    [[-0.0078, 3.9952, 4.0754], [2.9856, 4.6944, 5.0035], [3.2413, 4.7813, 4.9969]],
                ]
            )
        else:
            expected_slice = tf.convert_to_tensor(
                [
                    [[-4.8960, -2.3688, -3.0355], [-2.8478, -0.9836, -1.7418], [-2.9449, -1.3332, -2.1456]],
                    [[-5.8081, -3.4124, -4.1006], [-3.8561, -2.2081, -3.0323], [-3.8365, -2.4601, -3.3669]],
                    [[-0.0309, 3.9868, 4.0540], [2.9640, 4.6877, 4.9976], [3.2081, 4.7690, 4.9942]],
                ]
            )

        tf.debugging.assert_near(logits[0, :3, :3, :3], expected_slice, atol=1e-4)

    @slow
    def test_post_processing_semantic_segmentation(self):
        model = TFBeitForSemanticSegmentation.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640", from_pt=True
        )

        image_processor = BeitImageProcessor(do_resize=True, size=640, do_center_crop=False)

        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)
        outputs.logits = tf.stop_gradient(outputs.logits)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(500, 300)])
        expected_shape = tf.convert_to_tensor([500, 300])
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = tf.convert_to_tensor([160, 160])
        self.assertEqual(segmentation[0].shape, expected_shape)
