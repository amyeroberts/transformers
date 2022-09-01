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
"""Image processor class for ImageGPT."""

from typing import Iterable, List, Optional, Union

import numpy as np
import PIL.Image

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import get_resize_output_image_size, normalize, rescale, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    is_batched,
    to_numpy_array,
    valid_images,
)
from ...utils import logging


logger = logging.get_logger(__name__)


def squared_euclidean_distance(a, b):
    b = b.T
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=0)
    ab = np.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance(x, clusters)
    return np.argmin(d, axis=1)


class ImageGPTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ImageGPT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Set the class default for `do_normalize`. Controls whether to normalize the image.
        size (`int` *optional*, defaults to 256):
            Set the class default for the `size` parameter. Size of the image.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
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
        size: int = 256,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        image: np.ndarray,
        size: Union[int, Iterable[int]],
        crop_pct: Optional[float] = None,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize an image.

        # FIXME If size is an int, then the image is resized to (size, size). If size is an iterable of length 2, then
        the image is resized to (size[0], size[1]).

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`int` or `Iterable[int]`):
                Size of the output image.
            crop_pct (`float`, *optional*, defaults to None):
                Percentage of the image that will be cropped from the center. If set, the image is resized
            resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        output_size = get_resize_output_image_size(image, size=size)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
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
        clusters: Optional[Union[int, List[int]]] = None,
        do_resize: bool = None,
        size: int = None,
        resample: PIL.Image.Resampling = None,
        do_center_crop: bool = None,
        crop_pct: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_color_quantize: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`int`, *optional*, defaults to `self.size`):
                Size of the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image.Resampling`,
                Only has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
                Percentage of the image to crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            do_color_quantize (`bool`, *optional*, defaults to `self.do_color_quantize`):
                Whether to color quantize the image.
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
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_color_quantize = do_color_quantize if do_color_quantize is not None else self.do_color_quantize
        clusters = clusters if clusters is not None else self.clusters
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        crop_pct = crop_pct if crop_pct is not None else self.crop_pct
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_center_crop and crop_pct is None:
            raise ValueError("Crop_pct must be specified if do_center_crop is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, crop_pct=crop_pct, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        if do_color_quantize:
            images = [to_channel_dimension_format(image, ChannelDimension.LAST) for image in images]
            # color quantize from (batch_size, height, width, 3) to (batch_size, height, width)
            images = np.array(images)
            images = color_quantize(images, clusters).reshape(images.shape[:-1])

            # flatten to (batch_size, height*width)
            batch_size = images.shape[0]
            images = images.reshape(batch_size, -1)

        data = {"input_ids": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
