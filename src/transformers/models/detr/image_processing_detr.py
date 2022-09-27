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

import io
import pathlib
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch

import jax.numpy as jnp
import scipy.special
import scipy.stats

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import (
    center_to_corners_format,
    corners_to_center_format,
    id_to_rgb,
    normalize,
    rescale,
    resize,
    rgb_to_id,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_jax_tensor,
    is_tf_tensor,
    is_torch_tensor,
    to_numpy_array,
    valid_coco_detection_annotations,
    valid_coco_panoptic_annotations,
    valid_images,
)
from ...utils import is_flax_available, is_tf_available, is_torch_available, is_vision_available
from ...utils.generic import ExplicitEnum, TensorType


if is_vision_available():
    import PIL


if TYPE_CHECKING:
    from .modeling_detr import DetrObjectDetectionOutput, DetrSegmentationOutput


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


SUPPORTED_ANNOTATION_FORMATS = (AnnotionFormat.COCO_DETECTION, AnnotionFormat.COCO_PANOPTIC)


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (:obj:`Tuple[int, int]`):
            The input image size.
        size (:obj:`int`):
            The desired output size.
        max_size (:obj:`int`, `optional`):
            The maximum allowed output size.
    """
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


def get_resize_output_image_size(
    image_size: Tuple[int, int], size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = None
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        image_size (:obj:`Tuple[int, int]`):
            The input image size.
        size (:obj:`int`):
            The desired output size.
        max_size (:obj:`int`, `optional`):
            The maximum allowed output size.
    """
    if isinstance(size, (list, tuple)):
        # FIXME - is the size in the configuration in (width, height) format?
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)


def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (:obj:`np.ndarray`): The array to convert.
    """
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
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


def get_pad_size(images: List[np.ndarray]) -> Tuple[int, int]:
    """
    Computes the padding size for a list of images, where the padding size is the maximum width and height across all
    images in a batch.
    """
    input_channel_dimension = infer_channel_dimension_format(images[0])

    if input_channel_dimension == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_channel_dimension == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_channel_dimension}")
    return (max_height, max_width)


def bottom_right_pad(
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
    Args:
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
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
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Args:
    Convert a COCO polygon annotation to a mask.
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
def prepare_coco_detection_annotation(image, target, return_segmentation_masks: bool = False):
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

    return new_target


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format [N, H, W] where N is the number of masks

    Returns:
        boxes: bounding boxes in format [N, 4] in xyxy format
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


def prepare_coco_panoptic_annotation(
    image: np.ndarray, target: Dict, masks_path: Union[str, pathlib.Path], return_masks: bool = True
) -> Dict:
    """
    Prepare a coco panoptic annotation for DETR.
    """
    image_height, image_width = get_image_size(image)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
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
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


def get_segmentation_image(
    masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False
):
    h, w = input_size
    final_h, final_w = target_size

    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    if m_id.shape[-1] == 0:
        # We didn't detect any mask :(
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        m_id = m_id.argmax(-1).reshape(h, w)

    if deduplicate:
        # Merge the masks corresponding to the same stuff class
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    seg_img = id_to_rgb(m_id)
    seg_img = resize(seg_img, (final_w, final_h), resample=PIL.Image.Resampling.NEAREST)
    return seg_img


def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    final_h, final_w = target_size
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    m_id = rgb_to_id(np_seg_img)
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area


def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = scipy.special.softmax(logits, axis=-1)
    labels = probs.argmax(-1, keepdims=True)
    scores = np.take_along_axis(probs, labels, axis=-1)
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


def post_process_panoptic_sample(
    out_logits: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    processed_size: Tuple[int, int],
    target_size: Tuple[int, int],
    is_thing_map: Dict,
    threshold=0.85,
) -> Dict:
    """
    Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions for a single sample.

    Args:
        out_logits (:obj:`torch.Tensor`):
            The logits for this sample.
        masks (:obj:`torch.Tensor`):
            The predicted segmentation masks for this sample.
        boxes (:obj:`torch.Tensor`):
            The prediced bounding boxes for this sample. The boxes are in the normalized format (center_x, center_y,
            width, height) and values between [0, 1], relative to the size the image (disregarding padding).
        processed_size (:obj:`Tuple[int, int]`):
            The processed size of the image (h, w), as returned by the preprocessing step i.e. the size after data
            augmentation but before batching.
        target_size (:obj:`Tuple[int, int]`):
            The target size of the image, (h, w) corresponding to the requested final size of the prediction.
        is_thing_map (:obj:`Dict`):
            A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.
        threshold (:obj:`float`, `optional`, defaults to 0.85):
            The threshold used to binarize the segmentation masks.
    """
    # we filter empty queries and detection below threshold
    scores, labels = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PIL.Image.Resampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # It may be that we have several predicted masks for the same stuff class.
    # In the following, we track the list of masks ids for each stuff class (they are merged later on)
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    for k, label in enumerate(cur_classes):
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # We filter out any mask that is too small
    if cur_classes.size() > 0:
        # We know filter empty masks as long as we find some
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        while filtered_small.any():
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        cur_classes = np.ones((1, 1), dtype=np.int64)

    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    del cur_classes

    with io.BytesIO() as out:
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    return predictions


def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold=0.5,
    resample=PIL.Image.Resampling.NEAREST,
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (:obj:`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (:obj:`Tuple[int, int]`):
            The original size of the input image.
        target_size (:obj:`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (:obj:`float`, `optional`, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (:obj:`PIL.Image.Resampling`, defaults to :obj:`PIL.Image.Resampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    new_annotation["size"] = target_size

    for key, value in annotation:
        if key == "boxes":
            boxes = value
            # FIXME - is having (w, h) rather than (h, w) correct here?
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes
        elif key == "area":
            area = value
            # FIXME - is having (w, h) rather than (h, w) correct here?
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area
        elif key == "masks":
            masks = value[:, None].astype(np.float32)
            masks = resize(masks, target_size, resample=resample)
            masks = masks[:, 0] > threshold
            new_annotation["masks"] = masks
        else:
            new_annotation[key] = value
    return new_annotation


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
            Set the class default for `do_pad`. Controls whether to pad the image to the largest image in a batch and
            create a pixel mask.
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

    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        return_segmentation_masks: bool = False,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        format: Optional[AnnotionFormat] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETR model.
        """
        format = format if format is not None else self.format

        if format == AnnotionFormat.COCO_DETECTION:
            target = prepare_coco_detection_annotation(image, target, return_segmentation_masks)
        elif format == AnnotionFormat.COCO_PANOPTIC:
            target = prepare_coco_panoptic_annotation(
                image, target, masks_path=masks_path, return_masks=return_segmentation_masks
            )
        else:
            raise ValueError(f"Format {format} is not supported.")
        return target

    def prepare(self, image, target, return_segmentation_masks=False, masks_path=None):
        warnings.warn(
            "The `prepare` method is deprecated and will be removed in a future version. "
            "Please use `prepare_annotation` instead. Note: the `prepare_annotation` method "
            "does not return the image anymore.",
        )
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return image, target

    def convert_coco_poly_to_mask(self, *args, **kwargs):
        warnings.warn("The `convert_coco_poly_to_mask` method is deprecated and will be removed in a future version. ")
        return convert_coco_poly_to_mask(*args, **kwargs)

    def prepare_coco_detection(self, *args, **kwargs):
        warnings.warn("The `prepare_coco_detection` method is deprecated and will be removed in a future version. ")
        return prepare_coco_detection_annotation(*args, **kwargs)

    def prepare_coco_panoptic(self, *args, **kwargs):
        warnings.warn("The `prepare_coco_panoptic` method is deprecated and will be removed in a future version. ")
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Union[int, Tuple[int, int]],
        max_size: int,
        resample=PIL.Image.BILINEAR,
        data_format=None,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or (h, w) tuple. If size is an int, smaller
        edge of the image will be matched to this number.
        """
        size = get_resize_output_image_size(get_image_size(image), size, max_size)
        image = resize(image, size=size, resample=resample, data_format=data_format)
        return image

    def resize_annotation(self, annotation, orig_size, size, resample=PIL.Image.NEAREST) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, size=size, resample=resample)

    def rescale(self, image: np.ndarray, rescale_factor: Union[float, int]) -> np.ndarray:
        """
        Rescale the image by the given factor.
        """
        return rescale(image, rescale_factor)

    def normalize(
        self, image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]
    ) -> np.ndarray:
        """
        Normalize the image with the given mean and standard deviation.
        """
        return normalize(image, mean=mean, std=std)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from [x0, y0, x1, y1] to [center_x, center_y, w, h] format.
        """
        return normalize_annotation(annotation, image_size=image_size)

    def pad(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        input_channel_dimension: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Args:
        Pad the bottom and right of the image with zeros to the output size.
            image (`np.ndarray`):
                Image to pad.
            output_size (`Tuple[int, int]`):
                Output size of the image.
            input_channel_dimension (`ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be inferred from the input image.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `None`):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        # FIXME - what should the pattern here be?
        # 1. pad pads an individual image. This matches the behavior of the other transforms e.g. resize.
        # 2. pad pads a batch of images and returns a mask. This matches the behavior of `pad` tokenizers
        # (doesn't accept same input as tokenizer call however)
        # Keep `pad_and_create_pixel_mask` for now
        return bottom_right_pad(
            image, output_size=output_size, input_channel_dimension=input_channel_dimension, data_format=data_format
        )

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        return_segmentation_masks: bool = False,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
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
        format: Optional[Union[str, AnnotionFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        **kwargs
    ) -> BatchFeature:
        if "pad_and_return_pixel_mask" in kwargs:
            warnings.warn(
                "The `pad_and_return_pixel_mask` argument is deprecated and will be removed in a future version, "
                "use `do_pad` instead.",
                FutureWarning,
            )
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

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
        format = self.format if format is None else format

        if do_resize is not None and (size is None or max_size is None):
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
        if annotations is not None:
            if format == AnnotionFormat.COCO_DETECTION and not valid_coco_detection_annotations(annotations):
                raise ValueError(
                    "Invalid COCO detection annotations. Annotations must a dict (single image) of list of dicts"
                    "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
                    "being a list of annotations in the COCO format."
                )
            elif format == AnnotionFormat.COCO_PANOPTIC and not valid_coco_panoptic_annotations(annotations):
                raise ValueError(
                    "Invalid COCO panoptic annotations. Annotations must a dict (single image) of list of dicts "
                    "(batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with "
                    "the latter being a list of annotations in the COCO format."
                )
            elif format not in SUPPORTED_ANNOTATION_FORMATS:
                raise ValueError(
                    f"Unsupported annotation format: {format} must be one of {SUPPORTED_ANNOTATION_FORMATS}"
                )

        if (
            masks_path is not None
            and format == AnnotionFormat.COCO_PANOPTIC
            and not isinstance(masks_path, (pathlib.Path, str))
        ):
            raise ValueError(
                "The path to the directory containing the mask PNG files should be provided as a"
                f" `pathlib.Path` or string object, but is {type(masks_path)} instead."
            )

        # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
        if annotations is not None:
            prepared_images = []
            prepared_annotations = []
            for image, target in zip(images, annotations):
                image, target = self.prepare_annotation(
                    image, target, format, return_segmentation_masks=return_segmentation_masks, masks_path=masks_path
                )
                prepared_images.append(image)
                prepared_annotations.append(target)
            images = prepared_images
            annotations = prepared_annotations
            del prepared_images, prepared_annotations

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        # transformations
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = [], []
                for image, target in zip(images, annotations):
                    orig_size = get_image_size(image)
                    resized_image = self.resize(image, size=size, max_size=max_size, resample=resample)
                    resized_annotation = self.resize_annotation(target, orig_size, get_image_size(resized_image))
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [self.resize(image, size=size, max_size=max_size, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]
            if annotations is not None:
                annotations = [
                    self.normalize_annotation(annotation, get_image_size(image))
                    for annotation, image in zip(annotations, images)
                ]

        if do_pad:
            # Pads images and returns their mask: {'pixel_values': ..., 'pixel_mask': ...}
            data = self._pad_and_create_pixel_mask(images, data_format=data_format)
        else:
            images = [to_channel_dimension_format(image, data_format) for image in images]
            data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs["labels"] = BatchFeature(data=annotations, tensor_type=return_tensors)

        return encoded_inputs

    def _pad_and_create_pixel_mask(
        self, pixel_values_list: List["torch.Tensor"], data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> Dict:
        pad_size = get_pad_size(pixel_values_list)
        padded_images = [
            self.pad(image=image, output_size=pad_size, data_format=data_format) for image in pixel_values_list
        ]
        masks = [make_pixel_mask(image=image, output_size=pad_size) for image in pixel_values_list]
        return {"pixel_values": padded_images, "pixel_mask": masks}

    def pad_and_create_pixel_mask(
        self,
        pixel_values_list: List["torch.Tensor"],
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.
            data_format (`str` or [`~utils.ChannelDimension`], *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format of the images.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).

        """
        data = self._pad_and_create_pixel_mask(pixel_values_list, data_format)
        return BatchFeature(data, tensor_type=return_tensors)

    # POSTPROCESSING METHODS
    # inspired by https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258
    def post_process_object_detection(
        self, outputs: "DetrObjectDetectionOutput", target_sizes: Union[TensorType, Iterable]
    ) -> Dict:
        # infer input framework and get function to cast the inputs back to the right framework
        to_input_framework = get_numpy_to_framework_fn(outputs.logits)

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("The number of images and target sizes do not match.")

        out_logits = to_numpy_array(outputs.logits)
        out_bbox = to_numpy_array(outputs.pred_boxes)
        target_sizes = to_numpy_array(target_sizes)

        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the (h. w) for each image in the batch")

        prob = scipy.special.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {
                "scores": to_input_framework(score),
                "labels": to_input_framework(label),
                "boxes": to_input_framework(bboxes),
            }
            for score, label, bboxes in zip(scores, labels, boxes)
        ]
        return results

    def post_process(self, *args, **kwargs):
        warnings.warn(
            "This method is deprecated and will be removed in a future version. Please use"
            " post_process_object_detection instead.",
            DeprecationWarning,
        )
        return self.post_process_object_detection(*args, **kwargs)

    def post_process_segmentation(
        self,
        outputs: "DetrSegmentationOutput",
        target_sizes: Union[TensorType, Iterable],
        threshold: float = 0.9,
        mask_threshold: float = 0.5,
    ) -> Dict:
        """
        Converts the output of [`DetrForSegmentation`] into image segmentation predictions. Only supports PyTorch.

        Parameters:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction.
            threshold (`float`, *optional*, defaults to 0.9):
                Threshold to use to filter out queries.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, and masks for an image
            in the batch as predicted by the model.
        """
        out_logits, raw_masks = outputs.logits, outputs.pred_masks
        target_sizes = to_numpy_array(target_sizes)

        def post_process_segmentation_sample(logits, masks, size, threshold, mask_threshold):
            # we filter empty queries and detection below threshold
            scores, labels = score_labels_from_class_probabilities(logits)
            keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

            cur_scores = scores[keep]
            cur_labels = labels[keep]
            cur_masks = masks[keep]
            cur_masks = resize(cur_masks[:, None], size, resample=PIL.Image.Resampling.BILINEAR)
            cur_masks = safe_squeeze(cur_masks, 1)
            cur_masks = (scipy.special.expit(cur_masks) > mask_threshold) * 1
            prediction = {"scores": cur_scores, "labels": cur_labels, "masks": cur_masks}
            return prediction

        predictions = []
        for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
            # we filter empty queries and detection below threshold
            predictions.append(
                post_process_segmentation_sample(cur_logits, cur_masks, tuple(size), threshold, mask_threshold)
            )
        return predictions

    # inspired by https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218
    def post_process_instance_segmentation(
        self,
        results: List[Dict],
        outputs: "DetrSegmentationOutput",
        orig_target_sizes: Union[TensorType, Iterable],
        max_target_sizes: Union[TensorType, Iterable],
        threshold=0.5,
    ):
        """
        Converts the output of [`DetrForSegmentation`] into instance segmentation predictions.

        Args:
            results (:obj:`List[Dict]`):
                Results list obtained from the `post_process_object_detection` method, to which "masks" results will be
                added.
            outputs (:obj:`DetrSegmentationOutput`):
                Raw outputs of the model
            orig_target_sizes:
                Tensor containing the size (h,w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes:
                Tensor containing the max target size (h,w) of each image of the batch.
            threshold (:obj:`float`, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary masks.
        """
        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError("The number of images and target sizes do not match.")

        max_target_sizes = to_numpy_array(max_target_sizes)
        orig_target_sizes = to_numpy_array(orig_target_sizes)

        max_h, max_w = list(max_target_sizes.max(0))
        outputs_masks = safe_squeeze(outputs.pred_masks, 2)
        outputs_masks = resize(outputs_masks, (max_h, max_w), resample=PIL.Image.Resample.BILINEAR)
        outputs_masks = scipy.special.expit(outputs_masks) > threshold

        def post_process_instance_segmentation_sample(result, mask, max_target_size, orig_target_size):
            img_h, img_w = max_target_size
            processed_result = {}
            for key, value in result.items():
                if key == "masks":
                    mask = mask[:, None, :img_h, :img_w]
                    mask = resize(mask.astype(np.float32), orig_target_size, resample=PIL.Image.NEAREST)
                    mask = mask.astype(np.uint8)
                else:
                    processed_result[key] = value
            return processed_result

        processed_results = []
        for result, mask, max_target_size, orig_target_size in zip(
            results, outputs_masks, max_target_sizes, orig_target_sizes
        ):
            processed = post_process_instance_segmentation_sample(result, mask, max_target_size, orig_target_size)
            processed_results.append(processed)
        return processed_results

    def post_process_instance(self, *args, **kwargs):
        warnings.warn(
            "post_process_instance is deprecated and will be removed in a future version. Please use"
            " post_process_instance_segmentation instead."
        )
        return self.post_process_instance_segmentation(*args, **kwargs)

    # inspired by https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L241 # FIXME
    def post_process_panoptic_segmentation(
        self,
        outputs: "DetrSegmentationOutput",
        processed_sizes: Union[np.ndarray, torch.Tensor, List[Tuple[int, int]], tf.Tensor, jnp.ndarray],
        target_sizes: Union[np.ndarray, torch.Tensor, List[Tuple[int, int]], tf.Tensor, jnp.ndarray],
        is_thing_map: Dict,
        threshold: float = 0.85,
    ) -> List[Dict]:
        """
        Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions.

        Args:
            outputs (:obj:`DetrSegmentationOutput`):
                Raw outputs of the model.
            processed_sizes (:
                obj:`torch.Tensor` or :obj:`np.ndarray` or :obj:`List[Tuple[int, int]]` or :obj:`tf.Tensor` or
                :obj:`jnp.ndarray`): The processed sizes of the images, as returned by the preprocessing step.
            target_sizes (:
                obj:`torch.Tensor` or :obj:`np.ndarray` or :obj:`List[Tuple[int, int]]` or :obj:`tf.Tensor` or
                :obj:`jnp.ndarray`): The target sizes of the images, as returned by the preprocessing step.
            is_thing_map (:obj:`Dict`):
                A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.
            threshold (:obj:`float`, `optional`, defaults to 0.85):
                The threshold used to binarize the segmentation masks.
        """
        # default to is_thing_map of COCO panoptic
        is_thing_map = is_thing_map if is_thing_map is not None else {i: i <= 90 for i in range(201)}
        target_sizes = target_sizes if target_sizes is not None else processed_sizes

        if len(processed_sizes) != len(target_sizes):
            raise ValueError(
                f"Number of processed and target sizes should be the same. Got {len(processed_sizes)} and"
                f" {len(target_sizes)} instead."
            )

        out_logits, raw_masks, raw_boxes = outputs.logits, outputs.pred_masks, outputs.pred_boxes

        if not (len(out_logits) == len(raw_masks) == len(target_sizes)):
            raise ValueError(
                f"Number of images ({len(out_logits)}), predicted masks ({len(raw_masks)}) and targets "
                f"({len(target_sizes)}) should be the same. "
            )

        # The output dict values are ints, str and bool: there is not framework dependant output. We convert
        # the input to numpy before processing without having to consider the input framework.
        out_logits = to_numpy_array(out_logits)
        raw_masks = to_numpy_array(raw_masks)
        raw_boxes = to_numpy_array(raw_boxes)

        preds = []
        for logits, masks, boxes, processed_size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            pred = post_process_panoptic_sample(
                logits, masks, boxes, processed_size, target_size, is_thing_map, threshold
            )
            preds.append(pred)
        return preds

    def post_process_panoptic(self, *args, **kwargs):
        warnings.warn(
            "post_process_panoptic is deprecated and will be removed in a future version. Please use"
            " post_process_panoptic_segmentation instead."
        )
        return self.post_process_panoptic_segmentation(*args, **kwargs)
