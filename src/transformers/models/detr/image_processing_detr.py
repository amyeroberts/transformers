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
"""Image processor class for DETR."""


import pathlib
from typing import Dict, Tuple, Union, List, Optional, Iterable, Any

if is_vision_available():
    import PIL

from ...feature_extraction_utils import BatchFeature

import numpy as np

from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, valid_images, valid_coco_panoptic_annotations, valid_coco_detection_annotations, is_batched

from ...utils.generic import ExplicitEnum

from ...image_processing_utils import BaseImageProcessor

from ...image_transforms import corners_to_center_format, center_to_corners_format, resize, rescale, normalize, to_channel_dimension_format, rgb_to_id
from ...image_utils import get_image_size, ChannelDimension, infer_channel_dimension_format, to_channel_dimension_format


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"

SUPPORTED_ANNOTATION_FORMATS = (AnnotionFormat.COCO_DETECTION, AnnotionFormat.COCO_PANOPTIC)


def normalize_annotation(annotation, image_height, image_width):
    norm_annotation = {}
    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            boxes = corners_to_center_format(boxes)
            # FIXME - check height width order
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
        else:
            norm_annotation[key] = value
    return norm_annotation


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_pad_size(images: List[np.ndarray]) -> List[int]:
    input_channel_dimension = infer_channel_dimension_format(images[0])

    if input_channel_dimension == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_channel_dimension == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_channel_dimension}")
    return (max_height, max_width)


def pad(
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


# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L33
def convert_coco_poly_to_mask(self, segmentations, height, width):
    """
    Convert a COCO polygon annotation to a mask.
    Args:
        segmentations (`List[List[float]]`):
            List of polygons, each polygon represented by a list of x-y coordinates.
        height (`int`):
            Height of the mask.
        width (`int`):
            Width of the mask.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks




# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50
def prepare_coco_detection(image, target, return_segmentation_masks: bool = False):
    """
    Convert the target in COCO format into the format expected by DETR.
    """
    image_height, image_width = get_image_size(image)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # Get all COCO annotations for the given image.
    annotations = target["annotations"]
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # for conversion to coco api
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    boxes = [obj["bbox"] for obj in annotations]
    # guard against no boxes via resizing
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]

    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints[keep]

    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return image, new_target


def prepare_coco_panoptic(image: np.ndarray, target: Dict, masks_path: Union[str, pathlib.Path], return_masks: bool = True) -> Dict:
    """
    Prepare a coco panoptic annotation for DETR.
    """
    image_height, image_width = get_image_size(image)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = np.asarray(
        [target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64
    )
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        new_target["boxes"] = masks_to_boxes(masks)

        labels = np.array([segment_info["category_id"] for segment_info in target["segments_info"]])
        labels = labels.astype(np.int64)

        new_target["class_labels"] = labels
        new_target["iscrowd"] = np.asarray([segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64)
        new_target["area"] = np.asarray([segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32)

    return image, new_target


class DetrImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`): # FIXME
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_resize` parameter. Controls whether to resize the image's (height, width)
            dimensions to the specified `size`.
        size (`int` *optional*, defaults to 800):
            Set the class default for the `size` parameter. Size of the image.
        max_size (`int` *optional*, defaults to 1333):
            Set the class default for the `max_size` parameter. Controls the largest size an input image can have,
            (otherwise it's capped).
        resample (`PIL.Image.Resampling`, *optional*, defaults to `PIL.Image.Resampling.BILINEAR`):
            Set the class default for `resample`. Defines the resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Set the class default for the `do_rescale` parameter. Controls whether to rescale the image by the
            specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Set the class default for `rescale_factor`. Defines the scale factor to use if rescaling the image.
        do_normalize:
            Set the class default for `do_normalize`. Controls whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Set the class default for `image_mean`. This is a float or list of floats of length of the number of
            channels for
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Image standard deviation.
        do_pad (`bool`, *optional*, defaults to `True`):
            Set the class default for `do_pad`. Controls whether to pad the image to the largest image in a batch and create
            a pixel mask.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        format: Union[str, AnnotionFormat] = AnnotionFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: int = 800,
        max_size: int = 1333,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_pad: bool = True,
        **kwargs
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad

    def pad_and_return_pixel_mask(self) -> bool:
        pass

    # FIXME - how to pass defaults?
    def resize(self, image, size, max_size, resample=PIL.Image.BILINEAR, data_format=None):

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            height, width = image_size
            if max_size is not None:
                min_original_size = float(min((height, width)))
                max_original_size = float(max((height, width)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (height <= width and height == size) or (width <= height and width == size):
                return height, width

            if width < height:
                ow = size
                oh = int(size * height / width)
            else:
                oh = size
                ow = int(size * width / height)
            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                # FIXME - is the size in the configuration in (width, height) format?
                return size
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        size = get_size(get_image_size(image), size, max_size)
        image = resize(image, size=size, resample=resample, data_format=data_format)
        return image






    def resize_annotation(self, annotation):
        pass

    def pad(
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
        return pad(
            image, output_size=output_size, input_channel_dimension=input_channel_dimension, data_format=data_format
        )

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional,
        return_segmentation_masks: bool = False,
        masks_path: Optional[pathlib.Path] = None,
        format: Optional[Union[str, AnnotionFormat]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[int] = None,
        max_size: Optional[int] = None,
        resample: Optional[PIL.Image.Resampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs
    ) -> BatchFeature:
        format = self.format if format is None else format
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        max_size = self.max_size if max_size is None else max_size
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_pad = self.do_pad if do_pad is None else do_pad

        if do_resize is not None and size is None or max_size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")

        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        if not is_batched(images):
            images = [images]
            annotations = [annotations] if annotations is not None else None

        if annotations is not None and len(images) != len(annotations):
            raise ValueError(
                f"The number of images ({len(images)}) and annotations ({len(annotations)}) do not match."
            )

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        format = AnnotionFormat(format)
        if format == AnnotionFormat.COCO_DETECTION and not valid_coco_detection_annotations(annotations):
            raise ValueError(
                """Invalid COCO detection annotations.
                Annotations must a dict (single image) of list of dicts (batch of images) with the following keys:
                ``image_id` and `annotations`, with the latter being a list of annotations in the COCO format.""")
        elif format == AnnotionFormat.COCO_PANOPTIC and not valid_coco_panoptic_annotations(annotations):
            raise ValueError(
                """Invalid COCO panoptic annotations.
                Annotations must a dict (single image) of list of dicts (batch of images) with the following keys:
                `image_id` and `segments_info`, with the latter being a list of annotations in the COCO format.""")
        elif format not in SUPPORTED_ANNOTATION_FORMATS:
            raise ValueError(f"Unsupported annotation format: {format} must be one of {SUPPORTED_ANNOTATION_FORMATS}")

        if (
            masks_path is not None
            and format == AnnotionFormat.COCO_PANOPTIC
            and not isinstance(masks_path, pathlib.Path)
        ):
            raise ValueError(
                f"The path to the directory containing the mask PNG files should be provided as a"
                f" `pathlib.Path` object, but is {type(masks_path)} instead.")

        # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
        if annotations is not None:
            pass

        # transformations
        if do_resize:
            images = [self.resize(image, size, max_size, resample) for image in images]
            if annotations is not None:
                annotations = [self.resize_annotation(annotation, size, max_size) for annotation in annotations]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]
            if annotations is not None:
                annotations = [
                    self.normalize_annotation(annotation, *get_image_size(image))
                    for annotation, image in zip(annotations, images)
                ]

        if do_pad:
            pad_size = get_pad_size(images)
            padded_images = [
                self.pad_image(image=image, output_size=pad_size, data_format=data_format) for image in images
            ]
            masks = [make_pixel_mask(image=image, output_size=pad_size) for image in images]
            data = {"pixel_values": padded_images, "pixel_mask": masks}
        else:
            data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs["labels"] = BatchFeature(data=annotations, tensor_type=return_tensors)

        return encoded_inputs
