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
"""Image processor class for Vilt."""

from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import normalize, rescale, resize, to_channel_dimension_format
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


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def pad_image(
    image: np.ndarray,
    output_size: Tuple[int, int],
    input_channel_dimension: Optional[ChannelDimension] = None,
    data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    """
    Pad the bottom and right of the image with zeros to make it up to the output size.
    """
    if input_channel_dimension is None:
        input_channel_dimension = infer_channel_dimension_format(image)

    output_height, output_width = output_size
    input_height, input_width = get_image_size(image)
    pad_bottom = output_height - input_height
    pad_right = output_width - input_width

    if input_channel_dimension == ChannelDimension.FIRST:
        padded_image = np.pad(image, [(0, 0), (0, pad_bottom), (0, pad_right)], mode="constant", constant_values=0)
    elif input_channel_dimension == ChannelDimension.LAST:
        padded_image = np.pad(image, [(0, pad_bottom), (0, pad_right), (0, 0)], mode="constant", constant_values=0)
    else:
        raise ValueError(f"Invalid channel dimension format: {input_channel_dimension}")

    if data_format is not None:
        padded_image = to_channel_dimension_format(padded_image, data_format)

    return padded_image


def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    input_height, input_width = get_image_size(image)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


def get_pad_size(images: List[np.ndarray]) -> List[int]:
    input_channel_dimension = infer_channel_dimension_format(images[0])

    if input_channel_dimension == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_channel_dimension == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_channel_dimension}")
    return (max_height, max_width)


def get_resize_output_image_size(
    input_image: np.ndarray, shorter: int = 800, longer: int = 1333, size_divisor: int = 32
) -> Tuple[int, int]:
    input_height, input_width = get_image_size(input_image)
    min_size, max_size = shorter, longer

    scale = min_size / min(input_height, input_width)

    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    return new_height, new_width


class ViltImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViLT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        size (`int` *optional*, defaults to 384):
            Set the class default for the `size` parameter. Resize the shorter side of the input to the given size.
            Should be an integer. The longer side will be limited to under int((1333 / 800) * size) while preserving
            the aspect ratio. Only has an effect if `do_resize` is set to `True`.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        size_divisor (`int`, *optional*, defaults to 32):
            Set the class default for `size_divisor`. The size by which to make sure both the height and width can be
            divided. Only has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Set the class default for `do_normalize`. Controls whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Set the class default for `image_mean`. This is a float or list of floats of length of the number of
            channels for
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Set the class default for `image_std`. Image standard deviation.
        do_pad (`bool`, *optional*, defaults to `True`):
            Set the class default for `do_pad`. Controls whether to pad the image to the (max_height, max_width) of the
            images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: int = 384,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        size_divisor: int = 32,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        **kwargs
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad

    def resize(
        self,
        image: np.ndarray,
        size: int,
        size_divisor: int = 32,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
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
            size_divisor (`int`):
                The image is resized to a size that is a multiple of this value.
            resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        longer = int(1333 / 800 * size)
        output_size = get_resize_output_image_size(image, shorter=size, longer=longer, size_divisor=size_divisor)
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

    def pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        input_channel_dimension: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Pad the bottom and right of the image with zeros to the output size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            output_size (`Tuple[int, int]`):
                Output size of the image.
            input_channel_dimension (`ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be inferred from the input image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return pad_image(
            image, output_size=output_size, input_channel_dimension=input_channel_dimension, data_format=data_format
        )

    def make_pixel_mask(self, image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

        Args:
            image (`np.ndarray`):
                Image to make the pixel mask for.
            output_size (`Tuple[int, int]`):
                Output size of the mask.
        """
        input_height, input_width = get_image_size(image)
        mask = np.zeros(output_size, dtype=np.int64)
        mask[:input_height, :input_width] = 1
        return mask

    def pad_and_create_pixel_mask(
        self,
        images: np.ndarray,
        return_tensors: Optional[Union[str, TensorType]],
        data_format: Optional[ChannelDimension] = None,
    ) -> BatchFeature:
        """
        Pad the bottom and right of the image with zeros to the output size and return a pixel mask.

        Args:
            image (`np.ndarray`):
                Image to pad.
            return_tensors (`str`, *optional*, defaults to `None`):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        FutureWarning(
            "This method is deprecated and will be removed in a future release. Use pad_image and make_pixel_mask"
            " instead."
        )
        pad_size = get_pad_size(images)
        padded_images = [
            self.pad_image(image=image, output_size=pad_size, data_format=data_format) for image in images
        ]
        masks = [self.make_pixel_mask(image=image, output_size=pad_size) for image in images]

        data = {"pixel_values": padded_images, "pixel_mask": masks}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: int = None,
        size_divisor: int = None,
        resample: PIL.Image.Resampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
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
            size (`int`, *optional*, defaults to `self.size`):
                Size of the image.
            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
                The image is resized to a size that is a multiple of this value.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image.Resampling`,
                Only has an effect if `do_resize` is set to `True`.
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
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to the (max_height, max_width) in the batch. If True, a pixel mask is also
                created and returned.
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
        size = size if size is not None else self.size
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad

        if not is_batched(images):  # FIXME - use tree.is_nested
            images = [images]

        if not valid_images(images):  # FIXME - make check in preprocess_img
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [
                self.resize(image=image, size=size, size_divisor=size_divisor, resample=resample) for image in images
            ]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        if do_pad:
            pad_size = get_pad_size(images)
            padded_images = [
                self.pad_image(image=image, output_size=pad_size, data_format=data_format) for image in images
            ]
            masks = [self.make_pixel_mask(image=image, output_size=pad_size) for image in images]
            data = {"pixel_values": padded_images, "pixel_mask": masks}
        else:
            data = {"pixel_data": images}

        return BatchFeature(data=data, tensor_type=return_tensors)
