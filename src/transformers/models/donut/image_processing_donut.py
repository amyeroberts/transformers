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
"""Image processor class for Donut."""

import warnings
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    get_resize_output_image_size,
    make_thumbnail,
    normalize,
    rescale,
    resize,
    rotate,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    to_numpy_array,
    valid_images,
)
from ...utils import logging


logger = logging.get_logger(__name__)

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])


class DonutImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Donut image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        image_size (`dict`, *optional*, defaults to `{"height": 2560, "width": 1920}`):
            Set the class default for the `image_size` parameter. Sets the desired size of the output image.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BILINEAR`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_thumbnail` parameter. Controls whether to make a thumbnail of the input
            image.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to rotate the input if the height is greater than width.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the input to `size`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        do_normalize:
            Set the class default for `do_normalize`. Controls whether to normalize the image.
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
        image_size: Optional[Dict[str, int]] = None,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> None:
        # The size in the DonutFeatureExtractor config was stored as (width, height).
        if "size" in kwargs:
            size = kwargs.pop("size")
            warnings.warn(
                "The `size` parameter is deprecated and will be removed in a future version. Please use `image_size`"
                " instead.",
                FutureWarning,
            )
            if image_size is None:
                width, height = size
                image_size = {"width": width, "height": height}

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.image_size = image_size if image_size is not None else {"height": 2560, "width": 1920}
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def align_long_axis(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Align the long axis of the image with the horizontal axis

        Args:
            image (`np.ndarray`):
                Image to align.
            size (`Tuple[int, int]`):
                Desired size of the output image in (height, width) format.
        """

        def should_rotate(image, output_size):
            in_h, in_w = get_image_size(image)
            out_h, out_w = (in_h, in_w) if output_size is None else output_size
            return (in_h > in_w and out_h > out_w) or (in_h < in_w and out_h < out_w)

        if should_rotate(image, size):
            image = rotate(image, angle=-90, expand=True)

        return image

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
                Size of the output image. If a tuple (height, width) is passed, the smallest value is taken as the
                size, if an int is passed size it taken directly. The image is resized so the smallest dimension
                matches the size, and the other dimension is scaled preserving the aspect ratio.
            resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = min(size) if isinstance(size, Iterable) else size
        output_size = get_resize_output_image_size(image, size=size, default_to_square=False)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def pad(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
        random_padding: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to a given size. If random_padding is False, the image is padded with an even border on all sides
        to make it up to output_size. If random_padding is True, the image is padded with a random border thickness on
        each side to make it up to output_size.
        """
        in_height, in_width = get_image_size(image)
        out_height, out_width = size

        delta_height = out_height - in_height
        delta_width = out_width - in_width

        if delta_height < 0 or delta_width < 0:
            raise ValueError("The output size must be larger than the input size.")

        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        input_data_format = infer_channel_dimension_format(image)
        if input_data_format == ChannelDimension.FIRST:
            padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        elif input_data_format == ChannelDimension.LAST:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise ValueError(f"Invalid channel dimension format: {input_data_format}")

        image = np.pad(image, padding, mode="constant", constant_values=0)
        return image

    def make_thumbnail(
        self, image: np.ndarray, size: Tuple[int, int], data_format: Optional[ChannelDimension] = None, **kwargs
    ) -> np.ndarray:
        """
        Make a thumbnail of an image. The image is resized to fit in a box of size (size[0], size[1]) while preserving
        the aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Tuple[int, int]`):
                Desired size of the output image in (height, width) format.
        """
        return make_thumbnail(image, size=size, data_format=data_format, **kwargs)

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
        do_resize: bool = None,
        image_size: Optional[Dict[str, int]] = None,
        resample: PIL.Image.Resampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            image_size (`Dict[str, int]`, *optional*, defaults to `self.image_size`):
                Size of the output image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image.Resampling`,
                Only has an effect if `do_resize` is set to `True`.
            do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
                Whether to make a thumbnail of the image of requested `size`.
            do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
                Whether to rotate the image to align the long axis with the horizontal axis.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to `size`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by. Only has an effect if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
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
        if "size" in kwargs and image_size is None:
            warnings.warn(
                "The `size` argument is deprecated and will be removed in a future version, use `image_size` instead.",
                FutureWarning,
            )
            old_size = kwargs.pop("size")
            width, height = old_size
            image_size = {"height": height, "width": width}

        do_resize = do_resize if do_resize is not None else self.do_resize
        image_size = image_size if image_size is not None else self.image_size
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        random_padding = random_padding if random_padding is not None else self.random_padding
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = (image_size["height"], image_size["width"])

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if image_size is None and (do_resize or do_thumbnail or do_pad):
            raise ValueError("`size` must be provided if `do_resize`, `do_thumbnail` or `do_pad` is set to `True`")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and standard deviation must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_align_long_axis:
            images = [self.rotate(image, size) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_thumbnail:
            images = [self.make_thumbnail(image=image, size=size) for image in images]

        if do_pad:
            images = [self.pad(image=image, size=size, random_padding=random_padding) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
