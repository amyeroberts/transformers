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

import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, Repository, delete_repo, set_access_token
from requests.exceptions import HTTPError
from transformers import AutoImageProcessor, GLPNImageProcessor
from transformers.testing_utils import TOKEN, USER, check_json_file_has_correct_format, get_tests_dir, is_staging_test


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


SAMPLE_IMAGE_PROCESSING_CONFIG_DIR = get_tests_dir("fixtures")


class ImageProcessingSavingTestMixin:
    def test_image_process_to_json_string(self):
        image_process = self.image_processing_class(**self.image_process_dict)
        obj = json.loads(image_process.to_json_string())
        for key, value in self.image_process_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_process_to_json_file(self):
        image_process_first = self.image_processing_class(**self.image_process_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_process.json")
            image_process_first.to_json_file(json_file_path)
            image_process_second = self.image_processing_class.from_json_file(json_file_path)

        self.assertEqual(image_process_second.to_dict(), image_process_first.to_dict())

    def test_image_process_from_and_save_pretrained(self):
        image_process_first = self.image_processing_class(**self.image_process_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_process_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_process_second = self.image_processing_class.from_pretrained(tmpdirname)

        self.assertEqual(image_process_second.to_dict(), image_process_first.to_dict())

    def test_init_without_params(self):
        image_process = self.image_processing_class()
        self.assertIsNotNone(image_process)


class ImageProcessorUtilTester(unittest.TestCase):
    @unittest.skip("No tiny model has a defined image processor yet")
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = []
        response_mock.raise_for_status.side_effect = HTTPError

        # Download this model to make sure it's in the cache.
        _ = GLPNImageProcessor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("transformers.utils.hub.requests.head", return_value=response_mock) as mock_head:
            _ = GLPNImageProcessor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
            # This check we did call the fake head request
            mock_head.assert_called()


@is_staging_test
class ImageProcessorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        set_access_token(TOKEN)
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-image-processor")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-image-processor-org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-image-processor")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        image_processor = GLPNImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(
                os.path.join(tmp_dir, "test-image-processor"), push_to_hub=True, use_auth_token=self._token
            )

            new_image_processor = GLPNImageProcessor.from_pretrained(f"{USER}/test-image-processor")
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_in_organization(self):
        image_processor = GLPNImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(
                os.path.join(tmp_dir, "test-image-processor-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_image_processor = GLPNImageProcessor.from_pretrained("valid_org/test-image-processor-org")
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_dynamic_image_processor(self):
        CustomImageProcessor.register_for_auto_class()
        image_processor = CustomImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Repository(tmp_dir, clone_from=f"{USER}/test-dynamic-image-processor", use_auth_token=self._token)
            image_processor.save_pretrained(tmp_dir)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(
                image_processor.auto_map,
                {"AutoImageProcessor": "custom_image_processing.CustomImageProcessor"},
            )
            # The code has been copied from fixtures
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_image_processing.py")))

            repo.push_to_hub()

        new_image_processor = AutoImageProcessor.from_pretrained(
            f"{USER}/test-dynamic-image-processor", trust_remote_code=True
        )
        # Can't make an isinstance check because the new_image_processor is from the CustomImageProcessor class of a dynamic module
        self.assertEqual(new_image_processor.__class__.__name__, "CustomImageProcessor")
