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
"""Image processor class for ViT."""

from typing import Iterable, List, Optional, Union

import numpy as np
import PIL.Image

from transformers.utils.generic import TensorType
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import get_resize_output_image_size, to_channel_dimension_format, normalize, resize, rescale
from ...image_utils import (
    ChannelDimension,
    is_batched,
    to_numpy_array,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    valid_images,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class ViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the specified
            scale `rescale_factor`.
        do_normalize:
            Set the class default for `do_normalize`. Controls whether to normalize the image.
        size (`int` *optional*, defaults to 224):
            Set the class default for the `size` parameter. Size of the image.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BILINEAR`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Set the class default for `image_mean`. This is a float or list of floats of length of the number of
            channels for
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Image standard deviation.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        size: int = 224,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
        rescale_factor: Union[int, float] = 1 / 255,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        image: np.ndarray,
        size: Union[int, Iterable[int]],
        resample: PIL.Image.Resampling = PIL.Image.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize an image.

        If size is an int, then the image is resized to (size, size). If size is an iterable of length 2, then the
        image is resized to (size[0], size[1]).

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`int` or `Iterable[int]`):
                Size of the output image.
            resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        output_size = get_resize_output_image_size(image, size=size)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def rescale(
        self, image: np.ndarray, scale: Union[int, float], data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs
    ):
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            image_mean (`float` or `List[float]`):
                Image mean.
            image_std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        do_rescale: bool = None,
        do_normalize: bool = None,
        resample: PIL.Image.Resampling = None,
        size: int = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image.Resampling`,
                Only has an effect if `do_resize` is set to `True`.
            size (`int`, *optional*, defaults to `self.size`):
                Size of the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_tensors (`str`, *optional*, defaults to `None`):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
