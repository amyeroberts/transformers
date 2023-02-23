#!/usr/bin/env python
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


import json
import logging
import os
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Callable

import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset
from datasets.arrow_dataset import dataset_to_tf

import transformers
from transformers import (
    DefaultDataCollator,
    PushToHubCallback,
    TFTrainingArguments,
    create_optimizer,
    TFViTMAEForPreTraining,
    HfArgumentParser,
    ViTImageProcessor,
    ViTMAEConfig,
)
# from transformers.keras_callbacks import KerasMetricCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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
        # FIXME
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
    mask_ratio: float = field(
        default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )  # FIXME


@dataclass
class CustomTrainingArguments(TFTrainingArguments):
    base_learning_rate: float = field(
        default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def center_crop(image, size):
    size = (size, size) if isinstance(size, int) else size
    orig_height, orig_width, _ = image.shape
    crop_height, crop_width = size
    top = (orig_height - orig_width) // 2
    left = (orig_width - crop_width) // 2
    image = tf.image.crop_to_bounding_box(image, top, left, crop_height, crop_width)
    return image


# Numpy and TensorFlow compatible version of PyTorch RandomResizedCrop. Code adapted from:
# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
def random_crop(image, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    height, width, _ = image.shape
    area = height * width
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.exp(np.random.uniform(*log_ratio))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = np.random.randint(0, height - h + 1)
            j = np.random.randint(0, width - w + 1)
            return image[i : i + h, j : j + w, :]

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    w = width if in_ratio < min(ratio) else int(round(height * max(ratio)))
    h = height if in_ratio > max(ratio) else int(round(width / min(ratio)))
    i = (height - h) // 2
    j = (width - w) // 2
    return image[i : i + h, j : j + w, :]


def random_resized_crop(image, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    size = (size, size) if isinstance(size, int) else size
    image = random_crop(image, scale, ratio)
    image = tf.image.resize(image, size)
    return image


def collate_fn(examples):
    pixel_values = tf.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


def prepare_tf_dataset(
        dataset: Dataset,
        collate_fn: Callable,
        collate_fn_args: Optional[Dict] = None,
        cols_to_retain: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        drop_remainder: bool = False,
        prefetch: bool = True,
):
    """
    Helper function which constructs a TensorFlow dataset from a given dataset.

    This is an adapted version of the `to_tf_dataset` method in the datasets library which
    will return a dict instead of a tuple when there's no label and a single feature.
    """

    if collate_fn is None:
        collate_fn = DefaultDataCollator(return_tensors="np")

    collate_fn_args = collate_fn_args if collate_fn_args is not None else {}
    if batch_size is not None:
        batch_size = min(len(dataset), batch_size)

    output_signature, columns_to_np_types = dataset._get_output_signature(
        dataset,
        collate_fn=collate_fn,
        collate_fn_args=collate_fn_args,
        cols_to_retain=cols_to_retain,
        batch_size=batch_size,
        num_test_batches=1
    )

    if cols_to_retain is None:
        cols_to_retain = list(output_signature.keys())

    tf_dataset = dataset_to_tf(
        dataset,
        cols_to_retain=cols_to_retain,
        collate_fn=collate_fn,
        collate_fn_args=collate_fn_args,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        output_signature=output_signature,
        columns_to_np_types=columns_to_np_types
    )

    if prefetch:
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Remove a reference to the open Arrow file on delete
    def cleanup_callback(ref):
        dataset.__del__()
        dataset._TF_DATASET_REFS.remove(ref)

    dataset._TF_DATASET_REFS.add(weakref.ref(tf_dataset, cleanup_callback))

    return tf_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        exit("Must specify at least one of --do_train, --do_eval or --do_predict!")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detecting last checkpoint.
    checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset.
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Load pretrained model and image processor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update({"mask_ratio": model_args.mask_ratio, "norm_pix_loss": model_args.norm_pix_loss})

    # create image processor
    if model_args.image_processor_name:
        image_processor = ViTImageProcessor.from_pretrained(model_args.image_processor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        image_processor = ViTImageProcessor()

    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    if "shortest_edge" in image_processor.size:
        image_size = image_processor.size["shortest_edge"]
    else:
        image_size = (image_processor.size["height"], image_processor.size["width"])

    def _train_transforms(image):
        image = tf.keras.utils.img_to_array(image)
        image = random_resized_crop(image, size=image_size)
        image = tf.image.random_flip_left_right(image)
        image /= 255.0
        image = (image - image_processor.image_mean) / image_processor.image_std
        image = tf.transpose(image, perm=[2, 0, 1])
        return image

    def _val_transforms(image):
        image = tf.keras.utils.img_to_array(image)
        image = tf.image.resize(image, size=image_size)
        image = center_crop(image, size=image_size)
        image /= 255.0
        image = (image - image_processor.image_mean) / image_processor.image_std
        image = tf.transpose(image, perm=[2, 0, 1])
        return image

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[image_column_name]
        ]
        return example_batch

    train_dataset = None
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            train_transforms,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # Set the validation transforms
        eval_dataset = eval_dataset.map(
            val_transforms,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    with training_args.strategy.scope():
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint

        # create model
        if model_args.model_name_or_path:
            model = TFViTMAEForPreTraining.from_pretrained(
                model_path,
                from_pt=bool(".bin" in model_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            logger.info("Training new model from scratch")
            model = TFViTMAEForPreTraining(config)
        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        if training_args.do_train:
            num_train_steps = int(len(train_dataset) * training_args.num_train_epochs)
            if training_args.warmup_steps > 0:
                num_warmpup_steps = int(training_args.warmup_steps)
            elif training_args.warmup_ratio > 0:
                num_warmpup_steps = int(training_args.warmup_ratio * num_train_steps)
            else:
                num_warmpup_steps = 0

            optimizer, _ = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmpup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )

            # Models for pretraining don't have labels and instead only pass in the pixel_values
            # input to the model. We use a custom collate function to remove the labels and
            # prepare_tf_dataset to create a tf.data.Dataset.
            train_dataset = prepare_tf_dataset(
                train_dataset,
                shuffle=True,
                batch_size=total_train_batch_size,
                collate_fn=collate_fn,
                drop_remainder=True,
            ).with_options(dataset_options)
        else:
            optimizer = None

        if training_args.do_eval:
            eval_dataset = prepare_tf_dataset(
                eval_dataset,
                shuffle=False,
                batch_size=total_eval_batch_size,
                collate_fn=collate_fn,
                drop_remainder=True,
            ).with_options(dataset_options)

        model.compile(optimizer=optimizer, jit_compile=training_args.xla, metrics=["accuracy"])

        push_to_hub_model_id = training_args.push_to_hub_model_id
        if not push_to_hub_model_id:
            if model_path is not None:
                model_name = model_path.split("/")[-1]
            else:
                model_name = "vit-mae"
            push_to_hub_model_id = f"{model_name}-finetuned-image-classification"

        model_card_kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "masked-auto-encoding",
            "dataset": data_args.dataset_name,
            "tags": ["masked-auto-encoding"],
        }
        if model_path is not None:
            model_card_kwargs["finetuned_from"] = model_path

        callbacks = []
        if training_args.push_to_hub:
            callbacks.append(
                PushToHubCallback(
                    output_dir=training_args.output_dir,
                    hub_model_id=push_to_hub_model_id,
                    hub_token=training_args.push_to_hub_token,
                    tokenizer=image_processor,
                    **model_card_kwargs,
                )
            )

        if training_args.do_train:
            history = model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )

        eval_metrics = {}
        if training_args.do_eval:
            n_eval_batches = len(eval_dataset)
            eval_predictions = model.predict(eval_dataset, steps=n_eval_batches)
            eval_sample_loss = eval_predictions.loss
            eval_metrics['avg_loss'] = eval_sample_loss.mean()
            logging.info(f"Average loss: {eval_metrics['avg_loss']:.3f}")

        if training_args.output_dir is not None:
            all_results = {f"eval_{k}": v for k, v in eval_metrics.items()}
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                f.write(json.dumps(all_results))

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
