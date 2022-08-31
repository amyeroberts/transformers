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
"""Image processor class for Beit."""

from typing import Iterable, List, Optional, Union

import numpy as np
import PIL.Image
import tree

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import center_crop, get_resize_output_image_size, normalize, rescale, resize, to_channel_dimension_format
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


class BeitImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DeiT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        size (`int` *optional*, defaults to 256):
            Set the class default for the `size` parameter. Size of the image.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Set the class default for `do_center_crop`. Controls whether to center crop the image. If the input
            size is smaller than `crop_size` along any edge, the image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to `224`):
            Set the class default for the `crop_size` parameter. Desired output size when applying center-cropping.
            Only has an effect if `do_center_crop` is set to `True`.Controls the size of the image's (height, width)
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
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Set the class default for `do_reduce_labels`. Controls whether or not to reduce all label values of
            segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself
            is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: int = 256,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: int = 224,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        **kwargs
    ) -> None:
        if 'reduce_labels' in kwargs:
            do_reduce_labels = kwargs.pop('reduce_labels')

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_reduce_labels = do_reduce_labels

    def resize(
        self,
        image: np.ndarray,
        size: Union[int, Iterable[int]],
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
            resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        output_size = get_resize_output_image_size(image, size=size)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Union[int, Iterable[int]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Center crop an image.

        If crop_size is an int, then the image is cropped to (crop_size, crop_size). If crop_size is an iterable of
        length 2, then the image is cropped to (crop_size[0], crop_size[1]). If the input size is smaller than
        `crop_size` along any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`int` or `Iterable[int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        return center_crop(image, size=crop_size, data_format=data_format, **kwargs)

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

    def reduce_label(self, label: ImageInput) -> np.ndarray:
        label = to_numpy_array(label)
        # Avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        return label

    def preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: int = None,
        resample: PIL.Image.Resampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # FIXME - should these checks go here?
        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample)

        if do_center_crop:
            image = self.center_crop(image=image, crop_size=crop_size)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format)
        return image

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: bool = None,
        size: int = None,
        resample: PIL.Image.Resampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
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
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image.Resampling`,
                Only has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`int`, *optional*, defaults to `self.crop_size`):
                Size of the image after center crop. If one edge the image is smaller than `crop_size`, it will be padded
                with zeros and then cropped
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
            do_reduce_lavels (`bool`, *optional*, defaults to `self.do_reduce_levels`):
                Whether or not to reduce all label values of segmentation maps by 1.
                Usually used for datasets where 0 is used for background, and background itself is not included
                in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
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
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        crop_size = crop_size if crop_size is not None else self.crop_size
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not is_batched(images):  # FIXME - use tree.is_nested
            images = [images]
            segmentation_maps = [segmentation_maps] if segmentation_maps is not None else None

        if not valid_images(images):  # FIXME - make check in preprocess_img
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        images = tree.map_structure(
            lambda img: self.preprocess_image(
                image=img,
                do_resize=do_resize,
                do_center_crop=do_center_crop,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
                resample=resample,
                size=size,
                rescale_factor=rescale_factor,
                crop_size=crop_size,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
            ),
            images
        )

        data = {"pixel_values": images}

        if segmentation_maps is not None:
            # reduce zero label if needed
            if do_reduce_labels:
                segmentation_maps = tree.map_structure(
                    lambda seg_map: self.reduce_label(seg_map), segmentation_maps
                )
            segmentation_maps = tree.map_structure(
                lambda segmentation_map: self.preprocess_image(
                    image=segmentation_map,
                    do_resize=do_resize,
                    resample=resample,
                    size=size,
                    do_center_crop=do_center_crop,
                    crop_size=crop_size,
                    do_rescale=False,
                    do_normalize=False,
                    # Segmentation maps don't have a channel dimension
                    data_format=None,
                ),
                segmentation_maps
            )
            # An extra axis is added if the segmentation map is in (height, width) format - # FIXME is it always of no channel dim?
            segmentation_maps = tree.map_structure(lambda seg_map: seg_map.squeeze().astype(np.int64), segmentation_maps)
            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)
