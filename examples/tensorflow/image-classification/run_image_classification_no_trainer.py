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
""" Finetuning any 🤗 Transformers model for image classification."""
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset, load_metric

from huggingface_hub import Repository

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    TFAutoModelForImageClassification,
    AutoConfig,
    DefaultDataCollator,
    HfArgumentParser,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
    CONFIG_MAPPING,
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    DefaultDataCollator,
    HfArgumentParser,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.utils.dummy_tf_objects import PushToHubCallback
from transformers.utils.versions import require_version
from transformers import DefaultDataCollator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF
import tensorflow as tf  # noqa: E402

logger = logging.getLogger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/tensorflow/image-classification/requirements.txt") #FIXME - find version needed
# MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_TYPES = []

# region Helper classes
class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)

class PushToHubCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.push_to_hub()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
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
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."}) #FIXME - dir or file?
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."}) #FIXME - dir or file?
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    train_val_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "Percent to split off of train for validation"}
    ) # FIXME - remove validation_split_percentage?
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
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
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/Tensorflow versions.
    send_example_telemetry("run_image_classification", model_args, data_args, framework="tensorflow")

    # # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # FIXME - set logging level
    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            task="image-classification",
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
            task="image-classification",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(training_args.output_dir).name, token=training_args.hub_token)
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

        with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    if last_checkpoint is None:  # FIXME - merge iwt logic above
        model_path = model_args.model_name_or_path
    else:
        model_path = last_checkpoint

    # Load pretrained model and feature extractor
    with training_args.strategy.scope():
        # If passed along, set the training seed now.
        if training_args.seed is not None:
            set_seed(training_args.seed)
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        model = TFAutoModelForImageClassification.from_pretrained(
            model_path,
            config=config,
            # cache_dir=model_args.cache_dir,
            # revision=model_args.model_revision,
            # ignore_mismatched_sizes=training_args.ignore_mismatched_sizes, # FIXME - add to training args
        )
        # Preprocessing the datasets
        # Define tf.image transforms to be applied to each image.
        def transform_train(img):
            img = tf.keras.utils.img_to_array(img)
            img /= 255
            img = (img - tf.constant(feature_extractor.image_mean)) / (feature_extractor.image_std)
            # FIMXE - add normalise
            # Add rnadom crop
            # Add resize
            # FIXME - add horizontal flip
            img = tf.image.resize(img, (feature_extractor.size, feature_extractor.size))
            img = tf.transpose(img, (2, 0, 1))
            return img

        def transform_val(img):
            img = tf.keras.utils.img_to_array(img)
            img /= 255
            img = (img - tf.constant(feature_extractor.image_mean)) / (feature_extractor.image_std)
            # FIMXE - add normalise
            # Add central crop
            img = tf.image.resize(img, (feature_extractor.size, feature_extractor.size))
            img = tf.transpose(img, (2, 0, 1))
            return img

        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch.""" # FIXME
            pixel_values = [
                transform_train(img.convert("RGB")) for img in example_batch["image"]
            ]
            return {'pixel_values': pixel_values, 'labels': example_batch['labels']}

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch.""" # FIXME
            pixel_values = [
                transform_val(img.convert("RGB")) for img in example_batch["image"]
            ]
            return {'pixel_values': pixel_values, 'labels': example_batch['labels']}

        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
        if data_args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        # Set the validation transforms
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

        data_collator = DefaultDataCollator(return_tensors="tf")

        train_set = train_dataset.to_tf_dataset(
            batch_size=training_args.per_device_train_batch_size, # FIXME
            columns=["labels", "image"],
            shuffle=True,
            collate_fn=data_collator,
            prefetch=True,
        )
        eval_set = eval_dataset.to_tf_dataset(
            batch_size=training_args.per_device_train_batch_size,
            columns=["labels", "image"],
            shuffle=False,
            collate_fn=data_collator,
            prefetch=True,
        )

        # Optimizer
        num_replicas = training_args.strategy.num_replicas_in_sync
        batches_per_epoch = len(train_dataset) // (num_replicas * training_args.per_device_train_batch_size)
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(training_args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )
        # FIXME
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        # Scheduler and math around the number of training steps.
        # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = len(train_set) #math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

        # # Figure out how many steps we should save the Accelerator states
        # if hasattr(args.checkpointing_steps, "isdigit"):
        #     checkpointing_steps = args.checkpointing_steps
        #     if args.checkpointing_steps.isdigit():
        #         checkpointing_steps = int(args.checkpointing_steps)
        # else:
        #     checkpointing_steps = None

        # We need to initialize the trackers we use, and also store our configuration.
        # We initialize the trackers only on main process because `accelerator.log`
        # only logs on main process and we don't want empty logs/runs on other processes.
        # if args.with_tracking:
        #     # if accelerator.is_main_process:
        #     experiment_config = vars(args)
        #     # TensorBoard cannot log Enums, need the raw value
        #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        #     # accelerator.init_trackers("image_classification_no_trainer", experiment_config)

        # Get the metric function
        metric = load_metric("accuracy")

        # Train!
        # total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(training_args.max_train_steps))#, disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        # if args.resume_from_checkpoint:
        #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
        #         # accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        #         # accelerator.load_state(args.resume_from_checkpoint)
        #         path = os.path.basename(args.resume_from_checkpoint)
        #     else:
        #         # Get the most recent checkpoint
        #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        #         dirs.sort(key=os.path.getctime)
        #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        #     # Extract `epoch_{i}` or `step_{i}`
        #     training_difference = os.path.splitext(path)[0]

        #     if "epoch" in training_difference:
        #         starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        #         resume_step = None
        #     else:
        #         resume_step = int(training_difference.replace("step_", ""))
        #         starting_epoch = resume_step // len(train_set)
        #         resume_step -= starting_epoch * len(train_set)

        # FIXME - add push to hub call back ?

        # if tf_data["train"] is not None:

        # FIXME add push to hub call back

        if train_set is not None:
            callbacks = []
            if training_args.push_to_hub:
                callbacks = [PushToHubCallback(output_dir=training_args.output_dir, tokenizer=feature_extractor)]
            else:
                callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
            history = model.fit(
                train_set,
                validation_data=eval_set,
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )
        elif eval_dataset is not None:
            # If there's a validation dataset but no training set, just evaluate the metrics
            logger.info("Computing metrics on validation data...")
            loss, accuracy = model.evaluate(eval_dataset)
            logger.info(f"Loss: {loss:.5f}, Accuracy: {accuracy * 100:.4f}%")

    #     if args.push_to_hub and epoch < args.num_train_epochs - 1:
    #         # accelerator.wait_for_everyone()
    #         # unwrapped_model = accelerator.unwrap_model(model)
    #         unwrapped_model.save_pretrained(
    #             args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #         )
    #         # if accelerator.is_main_process:
    #         feature_extractor.save_pretrained(args.output_dir)
    #         repo.push_to_hub(
    #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
    #         )

    #     if args.checkpointing_steps == "epoch":
    #         output_dir = f"epoch_{epoch}"
    #         if args.output_dir is not None:
    #             output_dir = os.path.join(args.output_dir, output_dir)
    #         accelerator.save_state(output_dir)

    # if args.output_dir is not None:
    #     feature_extractor.save_pretrained(args.output_dir)
    #     if args.push_to_hub:
    #         repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    # if args.output_dir is not None:
    #     with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    #         json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)

    if training_args.push_to_hub:
        # You'll probably want to include some of your own metadata here!
        model.push_to_hub()



if __name__ == "__main__":
    main()
