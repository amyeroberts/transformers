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
"""Image processor class for Flava."""

import math
import random
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

import tree
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
from ...image_utils import ChannelDimension, ImageInput, is_batched, to_numpy_array, valid_images
from ...utils import logging


logger = logging.get_logger(__name__)


# These values are taken from CLIP
FLAVA_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FLAVA_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
FLAVA_CODEBOOK_MEAN = [0.0, 0.0, 0.0]
FLAVA_CODEBOOK_STD = [1.0, 1.0, 1.0]
LOGIT_LAPLACE_EPS: float = 0.1


# Inspired from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
        mask_group_max_aspect_ratio: float = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches

        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches

        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))

    def __repr__(self):
        repr_str = "MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.mask_group_min_patches,
            self.mask_group_max_patches,
            self.total_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                num_masked = mask[top : top + height, left : left + width].sum()
                # Overlap
                if 0 < height * width - num_masked <= max_mask_patches:
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.total_mask_patches:
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class FlavaImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Flava image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        size (`int` *optional*, defaults to 512):
            Set the class default for the `size` parameter. Size of the image.
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_center_crop` parameter. Controls whether to center crop the images
        crop_size (`int` *optional*, defaults to 224):
            Set the class default for the `crop_size` parameter. Size of the center crop.
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
        input_size_patches (`int`, *optional*, defaults to 14):
            Number of patches in the image in height and width direction. 14x14 = 196 total patches.
        total_mask_patches (`int`, *optional*, defaults to 75):
            Total number of patches that should be masked.
        mask_group_min_patches (`int`, *optional*, defaults to 16):
            Minimum number of patches that should be masked.
        mask_group_max_patches (`int`, *optional*, defaults to None):
            Maximum number of patches that should be masked.
        mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3):
            Minimum aspect ratio of the mask window.
        mask_group_max_aspect_ratio (`float`, *optional*, defaults to None):
            Maximum aspect ratio of the mask window
        codebook_do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input for codebook to a certain `codebook_size`.
        codebook_size (`int`, *optional*, defaults to 224):
            Resize the input for codebook to the given size. Only has an effect if `codebook_do_resize` is set to
            `True`.
        codebook_resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
        codebook_do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input for codebook at the center. If the input size is smaller than
            `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped.
        codebook_crop_size (`int`, *optional*, defaults to 224):
            Desired output size for codebook input when applying center-cropping. Only has an effect if
            `codebook_do_center_crop` is set to `True`.
        codebook_do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`.
        codebook_image_mean (`Tuple[float, float, float]`, *optional*, defaults to `[0, 0, 0]`):
            The sequence of means for each channel, to be used when normalizing images for codebook.
        codebook_image_std (`Tuple[float, float, float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images for codebook.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: int = 512,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        # Mask related params
        return_image_mask: bool = False,
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: int = 16,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook related params
        return_codebook_pixels: bool = False,
        codebook_do_resize: bool = True,
        codebook_size: bool = 112,
        codebook_resample: int = PIL.Image.Resampling.LANCZOS,
        codebook_do_center_crop: bool = True,
        codebook_crop_size: int = 112,
        codebook_do_rescale: bool = True,
        codebook_rescale_factor: Union[int, float] = 1 / 255,
        codebook_do_map_pixels: bool = True,
        codebook_do_normalize: bool = True,
        codebook_image_mean: Tuple[float, float, float] = FLAVA_CODEBOOK_MEAN,
        codebook_image_std: Tuple[float, float, float] = FLAVA_CODEBOOK_STD,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else FLAVA_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else FLAVA_IMAGE_STD

        self.return_image_mask = return_image_mask
        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

        self.return_codebook_pixels = return_codebook_pixels
        self.codebook_do_resize = codebook_do_resize
        self.codebook_size = codebook_size
        self.codebook_resample = codebook_resample
        self.codebook_do_center_crop = codebook_do_center_crop
        self.codebook_crop_size = codebook_crop_size
        self.codebook_do_rescale = codebook_do_rescale
        self.codebook_rescale_factor = codebook_rescale_factor
        self.codebook_do_map_pixels = codebook_do_map_pixels
        self.codebook_do_normalize = codebook_do_normalize
        self.codebook_image_mean = codebook_image_mean
        self.codebook_image_std = codebook_image_std

    @lru_cache()
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    ) -> FlavaMaskingGenerator:
        return FlavaMaskingGenerator(
            input_size=input_size_patches,
            total_mask_patches=total_mask_patches,
            mask_group_min_patches=mask_group_min_patches,
            mask_group_max_patches=mask_group_max_patches,
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
        )

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

    def map_pixels(self, image: np.ndarray) -> np.ndarray:
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

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
        # Mask related params
        return_image_mask: Optional[bool] = None,
        input_size_patches: Optional[int] = None,
        total_mask_patches: Optional[int] = None,
        mask_group_min_patches: Optional[int] = None,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: Optional[float] = None,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook related params
        return_codebook_pixels: Optional[bool] = None,
        codebook_do_resize: Optional[bool] = None,
        codebook_size: Optional[bool] = None,
        codebook_resample: Optional[int] = None,
        codebook_do_center_crop: Optional[bool] = None,
        codebook_crop_size: Optional[int] = None,
        codebook_do_rescale: Optional[bool] = None,
        codebook_rescale_factor: Optional[float] = None,
        codebook_do_map_pixels: Optional[bool] = None,
        codebook_do_normalize: Optional[bool] = None,
        codebook_image_mean: Optional[Iterable[float]] = None,
        codebook_image_std: Optional[Iterable[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            segmentation_maps (`ImageInput`, *optional*, defaults to `None`):
                Segmentation map to preprocess.
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
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
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
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255.
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
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        crop_size = crop_size if crop_size is not None else self.crop_size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        return_image_mask = return_image_mask if return_image_mask is not None else self.return_image_mask
        input_size_patches = input_size_patches if input_size_patches is not None else self.input_size_patches
        total_mask_patches = total_mask_patches if total_mask_patches is not None else self.total_mask_patches
        mask_group_min_patches = (
            mask_group_min_patches if mask_group_min_patches is not None else self.mask_group_min_patches
        )
        mask_group_max_patches = (
            mask_group_max_patches if mask_group_max_patches is not None else self.mask_group_max_patches
        )
        mask_group_min_aspect_ratio = (
            mask_group_min_aspect_ratio
            if mask_group_min_aspect_ratio is not None
            else self.mask_group_min_aspect_ratio
        )
        mask_group_max_aspect_ratio = (
            mask_group_max_aspect_ratio
            if mask_group_max_aspect_ratio is not None
            else self.mask_group_max_aspect_ratio
        )

        return_codebook_pixels = (
            return_codebook_pixels if return_codebook_pixels is not None else self.return_codebook_pixels
        )
        codebook_do_resize = codebook_do_resize if codebook_do_resize is not None else self.codebook_do_resize
        codebook_size = codebook_size if codebook_size is not None else self.codebook_size
        codebook_resample = codebook_resample if codebook_resample is not None else self.codebook_resample
        codebook_do_rescale = codebook_do_rescale if codebook_do_rescale is not None else self.codebook_do_rescale
        codebook_rescale_factor = (
            codebook_rescale_factor if codebook_rescale_factor is not None else self.codebook_rescale_factor
        )
        codebook_do_center_crop = (
            codebook_do_center_crop if codebook_do_center_crop is not None else self.codebook_do_center_crop
        )
        codebook_crop_size = codebook_crop_size if codebook_crop_size is not None else self.codebook_crop_size
        codebook_do_map_pixels = (
            codebook_do_map_pixels if codebook_do_map_pixels is not None else self.codebook_do_map_pixels
        )
        codebook_do_normalize = (
            codebook_do_normalize if codebook_do_normalize is not None else self.codebook_do_normalize
        )
        codebook_image_mean = codebook_image_mean if codebook_image_mean is not None else self.codebook_image_mean
        codebook_image_std = codebook_image_std if codebook_image_std is not None else self.codebook_image_std

        if not is_batched(images):  # FIXME - use tree.is_nested
            images = [images]

        if not valid_images(images):  # FIXME - make check in preprocess_img
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        processed_images = tree.map_structure(
            lambda img: self.preprocess_image(
                image=img,
                do_resize=do_resize,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
                resample=resample,
                size=size,
                rescale_factor=rescale_factor,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
            ),
            images,
        )

        data = {"pixel_values": processed_images}

        if return_codebook_pixels:
            codebook_images = tree.map_structure(
                lambda img: self.preprocess_image(
                    image=img,
                    do_resize=codebook_do_resize,
                    do_center_crop=codebook_do_center_crop,
                    do_rescale=codebook_do_rescale,
                    do_normalize=codebook_do_normalize,
                    resample=codebook_resample,
                    size=codebook_size,
                    crop_size=codebook_crop_size,
                    rescale_factor=codebook_rescale_factor,
                    image_mean=codebook_image_mean,
                    image_std=codebook_image_std,
                    data_format=data_format,
                ),
                images,
            )
            data["codebook_pixel_values"] = codebook_images

        if return_image_mask:
            mask_generator = self.masking_generator(
                input_size_patches=input_size_patches,
                total_mask_patches=total_mask_patches,
                mask_group_min_patches=mask_group_min_patches,
                mask_group_max_patches=mask_group_max_patches,
                mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
                mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
            )
            masks = [mask_generator() for _ in images]
            data["bool_masked_pos"] = masks

        return BatchFeature(data=data, tensor_type=return_tensors)
