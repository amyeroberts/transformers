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

from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import PIL

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
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
    ImageInput,
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
        size (`Dict[str, int]`, *optional*, defaults to {"shortest_edge": 224}):
            Desired image size after `resize`. If size is a dict with keys "width" and "height", the image will be
            resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value `c` is
            rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value i.e, if
            height > width, then image will be rescaled to (size * height / width, size).
        resample (`PIL.Image` resampling filter, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether or not to center crop the input to `size`.
        crop_size (`Dict`, *optional*, defaults to {"height": 224, "width": 224}):
            Desired image size after `center_crop`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample=PIL.Image.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_MEAN,
        image_std: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_STD,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample=PIL.Image.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize an image.

        If size is a dict with keys "width" and "height", the image will be resized to (height, width).

        If size is a dict with key "shortest_edge", the shortest edge value `c` is rescaled to int(`c` * (256/224)).
        The smaller edge of the image will be matched to this value i.e, if height > width, then image will be rescaled
        to (size * height / width, size).

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image
                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value
                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value
                i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`PIL.Image` resampling filter, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size_dict = get_size_dict(size, default_to_square=False)
        # size_dict is a dict with either keys "height" and "width" or "shortest_edge"
        if "shortest_edge" in size:
            shortest_edge = int((256 / 224) * size["shortest_edge"])
            output_size = get_resize_output_image_size(image, size=shortest_edge, default_to_square=False)
            size_dict = {"height": output_size[0], "width": output_size[1]}
        return resize(
            image, size=(size_dict["height"], size_dict["width"]), resample=resample, data_format=data_format, **kwargs
        )

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Center crop an image.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Dict {"height": int, "width": int} specifying the size of the output image after cropping.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size)
        return center_crop(image, size=(size["height"], size["width"]), data_format=data_format, **kwargs)

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
            mean (`float` or `List[float]`):
                Image mean.
            std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        image_std: Optional[Union[float, Iterable[float]]] = None,
        return_tensors: Optional[TensorType] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images to be used as input to a LeViT model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image
                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value
                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value
                i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`PIL.Image` resampling filter, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the output image after center cropping. Crops images to (crop_size["height"],
                crop_size["width"]).
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values by `rescaling_factor` - typical to values between 0 and 1.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Factor to rescale the image pixel values by.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image pixel values by `image_mean` and `image_std`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to normalize the image pixel values by.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to normalize the image pixel values by.
            return_tensors (`str` or `TensorType`, *optional*, defaults to `None`):
                Return type of the output. `None` returns a list of `np.ndarray`. `"np"` returns a `np.ndarray`. `"tf"`
                returns a `tf.Tensor`. `"pt"` returns a `pt.Tensor`. `"jax"` returns a `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. If `None`, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size)

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
            images = [self.center_crop(image, crop_size) for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
