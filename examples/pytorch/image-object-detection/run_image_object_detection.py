#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    ConvertBoundingBoxFormat,
    Normalize,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


""" Fine-tuning a 🤗 Transformers model for image object detection"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-object-detection/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_OBJECT_DETECTION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = [
        {
            # Take the augmented bboxes
            "boxes": torch.tensor(example["bboxes"]),
            "class_labels": torch.tensor(example["labels"]),
        }
        for example in examples
    ]
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_object_detection", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'object-detection' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].info.features["objects"].feature["category"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="object-detection",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _train_transforms = Compose(
        [
            # ToTensor(),
            # RandomResizedCrop(size),
            # RandomHorizontalFlip(),
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            # ToTensor(),
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    from torchvision import datapoints
    from torchvision.transforms.v2 import functional as F

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        batch_pixel_values = []
        batch_bboxes = []
        batch_labels = []

        pil_images = [pil_image.convert("RGB") for pil_image in example_batch["image"]]
        pil_images = [F.to_tensor(pil_image) for pil_image in pil_images]
        pil_images = [datapoints.Image(pil_image) for pil_image in pil_images]
        bboxes = [obj["bbox"] for obj in example_batch["objects"]]
        labels = [obj["category"] for obj in example_batch["objects"]]
        # FIxME - double check format
        bboxes = [
            datapoints.BoundingBox(
                bbox, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=F.get_spatial_size(img)
            )
            for img, bbox in zip(pil_images, bboxes)
        ]
        bboxes = [ConvertBoundingBoxFormat(datapoints.BoundingBoxFormat.CXCYWH)(bbox) for bbox in bboxes]

        for pil_img, img_bboxes, label in zip(pil_images, bboxes, labels):
            pixel_values, img_bboxes, label = _train_transforms(pil_img, img_bboxes, label)
            # Convert the bbox to relative coordinates
            # We can assume all the images as the same dimensiona after transformation because they are now batched.
            bbox, (bbox_height, bbox_width) = img_bboxes.data, img_bboxes.spatial_size
            bbox[:, 0] = bbox[:, 0] / bbox_width
            bbox[:, 1] = bbox[:, 1] / bbox_height
            bbox[:, 2] = bbox[:, 2] / bbox_width
            bbox[:, 3] = bbox[:, 3] / bbox_height
            batch_pixel_values.append(pixel_values)
            batch_bboxes.append(bbox)
            batch_labels.append(label)

        example_batch["pixel_values"] = torch.stack(batch_pixel_values)
        example_batch["bboxes"] = batch_bboxes
        example_batch["labels"] = batch_labels
        return example_batch

    def val_transforms(example_batch):
        # """Apply _val_transforms across a batch."""
        # example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        # img_size = example_batch["pixel_values"][0].shape[-2:]
        # # example_batch["pixel_mask"] = [torch.ones(img_size) for _ in example_batch["pixel_values"]]
        # return example_batch

        batch_pixel_values = []
        batch_bboxes = []
        batch_labels = []

        pil_images = [pil_image.convert("RGB") for pil_image in example_batch["image"]]
        pil_images = [F.to_tensor(pil_image) for pil_image in pil_images]
        pil_images = [datapoints.Image(pil_image) for pil_image in pil_images]
        bboxes = [obj["bbox"] for obj in example_batch["objects"]]
        labels = [obj["category"] for obj in example_batch["objects"]]
        # FIxME - double check format
        bboxes = [
            datapoints.BoundingBox(
                bbox, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=F.get_spatial_size(img)
            )
            for img, bbox in zip(pil_images, bboxes)
        ]
        bboxes = [ConvertBoundingBoxFormat(datapoints.BoundingBoxFormat.CXCYWH)(bbox) for bbox in bboxes]

        for pil_img, img_bboxes, label in zip(pil_images, bboxes, labels):
            pixel_values, img_bboxes, label = _train_transforms(pil_img, img_bboxes, label)
            # Convert the bbox to relative coordinates
            # We can assume all the images as the same dimensiona after transformation because they are now batched.
            bbox, (bbox_height, bbox_width) = img_bboxes.data, img_bboxes.spatial_size
            bbox[:, 0] = bbox[:, 0] / bbox_width
            bbox[:, 1] = bbox[:, 1] / bbox_height
            bbox[:, 2] = bbox[:, 2] / bbox_width
            bbox[:, 3] = bbox[:, 3] / bbox_height
            batch_pixel_values.append(pixel_values)
            batch_bboxes.append(bbox)
            batch_labels.append(label)

        example_batch["pixel_values"] = torch.stack(batch_pixel_values)
        example_batch["bboxes"] = batch_bboxes
        example_batch["labels"] = batch_labels
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "object-detection",
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
