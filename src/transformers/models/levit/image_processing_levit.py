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
"""Image processor class for LeViT."""

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    center_crop,
    get_resize_output_image_size,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    is_batched,
    to_numpy_array,
    valid_images,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class LevitImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LeViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the shortest edge of the
            input to int(256/224 *`size`).
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether or not to center crop the input to `size`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then shorter side of input will be resized to 'size'.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        image_mean (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        size: int = 224,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        rescale_factor: Union[int, float] = 1 / 255,
        image_mean: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_MEAN,
        image_std: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_STD,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

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

        If size is an int `c`, the value is rescaled to int(`c` * (256/224)). The smaller edge of the image will be
        matched to this value i.e, if height > width, then image will be rescaled to (size * height / width, size). If
        size is a iterable of length 2 (h, w), then the image is resized to (h, w).

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
        if isinstance(size, int):
            output_size = int((256 / 224) * size)
            output_size = get_resize_output_image_size(image, size=output_size, default_to_square=False)
        else:
            output_size = size
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def center_crop(
        self,
        image: np.ndarray,
        size: Union[int, Tuple[int, int]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Center crop an image.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Tuple[int, int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = (size, size) if isinstance(size, int) else size
        return center_crop(image, size=size, data_format=data_format, **kwargs)

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
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
        images,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        do_center_crop: Optional[bool] = None,
        size=None,
        resample=None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        image_std: Optional[Union[float, Iterable[float]]] = None,
        return_tensors: Optional[TensorType] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> BatchFeature:
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample

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
            images = [self.resize(image, size, resample) for image in images]

        if do_center_crop:
            images = [self.center_crop(image, size) for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
