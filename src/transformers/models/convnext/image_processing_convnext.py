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
"""Image processor class for ConvNeXT."""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL.Image

import tree
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


class ConvNextImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ConvNeXT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 384}`):
            Set the class default for the `size` parameter. Controls the resolution of the output image after `resize`
            is applied. If `size["shortest_edge"]` >= 384, the image is resized to (`size["shortest_edge"]`,
            `size["shortest_edge"]`). Otherwise, the smaller edge of the image will be matched to
            int(`size["shortest_edge"]`/`crop_pct`), after which the image is cropped to `(size["shortest_edge"],
            size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
        crop_pct (`float` *optional*, defaults to 244 / 256):
            Set the class default for the `crop_pct` parameter. The percentage of the image to crop. Only has an effect
            if `do_resize` is `True` and size < 384.
        resample (`PIL.Image` resampling filter, *optional*, defaults to `PIL.Image.BILINEAR`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
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
            Image standard deviation.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: int = 384,
        crop_pct: float = None,
        resample=PIL.Image.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 384}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        # Default value set here for backwards compatibility where the value in config is None
        self.crop_pct = crop_pct if crop_pct is not None else 224 / 256
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Center crop an image to (`size["height"]`, `size["width"]`).

        If the input size is smaller than `size` along any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`int` or `Iterable[int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=True)
        return center_crop(image, size=(size["height"], size["width"]), data_format=data_format, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        crop_pct: float,
        resample=PIL.Image.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form {"shortest_edge": int}, specifying the size of the output image. If
                size["shortest_edge" >= 384 image is resized to (`size["shortest_edge"], `size["shortest_edge"]`).
                Otherwise, the smaller edge of the image will be matched to int(`size["shortest_edge"]`/`crop_pct`),
                after which the image is cropped to (`size["shortest_edge"], `size["shortest_edge"]`).
            crop_pct (`float`):
                Percentage of the image to crop. Only has an effect if size < 384.
            resample (`PIL.Image` resampling filter, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=False)
        shortest_edge = size["shortest_edge"]

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(image, size=resize_shortest_edge, default_to_square=False)
            image = resize(image=image, size=resize_size, resample=resample, data_format=data_format, **kwargs)
            # then crop to (shortest_edge, shortest_edge)
            return center_crop(image=image, size=(shortest_edge, shortest_edge), data_format=data_format, **kwargs)
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return resize(
                image, size=(shortest_edge, shortest_edge), resample=resample, data_format=data_format, **kwargs
            )

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
            data_format (`str` or `ChannelDimension`, *optional*):
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
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        crop_pct: Optional[float] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_resize and size["shortest_edge"] < 384 and crop_pct is None:
            raise ValueError("crop_pct must be specified if size < 384.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if do_resize:
            image = self.resize(image=image, size=size, crop_pct=crop_pct, resample=resample)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std)

        image = to_channel_dimension_format(image, data_format)
        return image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        crop_pct: float = None,
        resample=None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
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
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
                is resized to (`size["shortest_edge"]`, `size["shortest_edge"]`). Otherwise, the smaller edge of the
                image will be matched to int(`size["shortest_edge"]`/`crop_pct`), after which the image is cropped to
                `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
            crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
                Percentage of the image to crop if size < 384.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of `PIL.Image`, resampling filters.
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
            return_tensors (`str`, *optional*):
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
        crop_pct = crop_pct if crop_pct is not None else self.crop_pct
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        images = tree.map_structure(
            lambda img: self.preprocess_image(
                image=img,
                do_resize=do_resize,
                size=size,
                crop_pct=crop_pct,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
            ),
            images,
        )

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
