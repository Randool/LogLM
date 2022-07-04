"""
@File        :  trainer.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/27
@Version     :  1.0
@Description :  None
"""
import logging
import math
import os
import sys
import time
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed, TrainerState, PretrainedConfig, is_torch_tpu_available, \
    __version__
from transformers.deepspeed import deepspeed_init, deepspeed_reinit
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.integrations import hp_params
from transformers.trainer_pt_utils import IterableDatasetShard, nested_detach
from transformers.trainer_utils import get_last_checkpoint, TrainOutput, has_length, ShardedDDPOption, HPSearchBackend, \
    speed_metrics, EvalLoopOutput

from dataset import DatasetForShuffledFile
from model import LogLMTokenizer, LogLMForPreTraining
from utils import LoggerCallback

# Name of the files used for checkpointing
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class LogLMTrainer(Trainer):
    def __init__(
            self,
            model: LogLMForPreTraining,
            args: TrainingArguments,
            tokenizer: LogLMTokenizer,
            train_data_files: List[str],
            eval_data_file: str,
            num_examples_for_epoch: List[int],
            logger: logging.Logger = None,
    ):
        assert len(train_data_files) == len(num_examples_for_epoch), \
            f"Length mismatch. $1={len(train_data_files)}, $2={len(num_examples_for_epoch)}"

        self.train_data_files = train_data_files
        self.num_examples_for_epoch = num_examples_for_epoch
        self.epoch_idx = 0

        self.train_dataset = DatasetForShuffledFile(train_data_files[self.epoch_idx],
                                                    mask_token_id=tokenizer.mask_token_id,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    num_examples=num_examples_for_epoch[self.epoch_idx])
        self.eval_dataset = DatasetForShuffledFile(eval_data_file,
                                                   mask_token_id=tokenizer.mask_token_id,
                                                   pad_token_id=tokenizer.pad_token_id)

        self.logger = logger if logger else logging.getLogger(__name__)

        super().__init__(
            model=model,
            args=args,
            data_collator=self.train_dataset.collate_fn,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=tokenizer,
            callbacks=[LoggerCallback(self.logger)],
        )

        self.label_names = ["labels", "sentence_pair_label"]
        self.loss_names = ["loss", "mlm_loss", "sop_loss", "dec_mlm_loss"]
        self.logits_names = ["prediction_logits", "dec_prediction_logits", "seq_relationship_logits"]

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        args = self.args
        dataset = self.train_dataset
        per_device_train_batch_size = args.per_gpu_train_batch_size or args.per_device_train_batch_size
        train_batch_size = args.train_batch_size

        # Shrink batch_size on last dataset
        if dataset.num_examples == min(self.num_examples_for_epoch):
            per_device_train_batch_size = max(1, per_device_train_batch_size // 16)
            train_batch_size = per_device_train_batch_size * max(1, args.n_gpu)

        if args.world_size > 1:
            dataset = IterableDatasetShard(
                dataset,
                batch_size=train_batch_size,
                drop_last=args.dataloader_drop_last,
                num_processes=args.world_size,
                process_index=args.process_index,
            )

        return DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )

    def get_next_train_dataloader(self):
        self.epoch_idx = (self.epoch_idx + 1) % len(self.train_data_files)

        self.train_dataset = DatasetForShuffledFile(
            self.train_data_files[self.epoch_idx],
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_examples=self.num_examples_for_epoch[self.epoch_idx]
        )

        return self.get_train_dataloader()

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Dict[str, Any] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            self.logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    self.logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can len() on the dataset or not
        train_dataset_is_sized = has_length(self.train_dataset)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size, but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs, so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        delay_optimizer_creation = (
                self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE or is_sagemaker_mp_enabled()
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {num_examples}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info(f"  Continuing training from epoch {epochs_trained}")
            self.logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                self.logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
                train_dataloader = self.get_next_train_dataloader()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        args.gradient_accumulation_steps >= steps_in_epoch == (step + 1)
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping
                        if self.do_grad_scaling:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                self.logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            train_dataloader = self.get_next_train_dataloader()

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        self.logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here, so we are sur the model has been saved by process 0.
            dist.barrier()

            self.logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                if self.deepspeed:
                    # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                    deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
                    self.model = deepspeed_engine.module
                    self.model_wrapped = deepspeed_engine
                    self.deepspeed = deepspeed_engine
                    self.optimizer = optimizer
                    self.lr_scheduler = lr_scheduler
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                    )
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
            else:
                self.logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_loss(self, model: LogLMForPreTraining, inputs, return_outputs=False):
        outputs = model.forward(**inputs, return_dict=True, label_smoother_epsilon=self.args.label_smoothing_factor)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def evaluate(
            self,
            eval_dataset: Optional[DatasetForShuffledFile] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg, but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        self.logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            self.logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            self.logger.info("  Num examples: Unknown")
        self.logger.info(f"  Batch size = {dataloader.batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        observed_num_examples, steps = 0, 1e-8
        loss_sum, mlm_loss_sum, sop_loss_sum, dec_mlm_loss_sum = 0, 0, 0, 0
        mlm_acc_sum, dec_mlm_acc_sum, sop_acc_sum = 0, 0, 0

        for inputs in dataloader:
            observed_batch_size = inputs["input_ids"].shape[0]
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
            steps += 1

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            loss, mlm_loss, sop_loss, dec_mlm_loss = losses

            loss_sum += loss.sum().item()
            mlm_loss_sum += mlm_loss.sum().item()
            sop_loss_sum += sop_loss.sum().item()
            dec_mlm_loss_sum += dec_mlm_loss.sum().item()

            # Compute Metrics
            (mlm_logits, dec_mlm_logits, sop_logits), (mlm_labels, sop_labels) = logits, labels
            mask_ids = inputs["input_ids"].eq(inputs["mask_token_id"])
            dec_mask = mlm_labels.ne(-100)

            mlm_pred = mlm_logits.argmax(dim=-1)
            dec_mlm_pred = dec_mlm_logits.argmax(dim=-1)
            sop_pred = sop_logits.argmax(dim=-1)

            mlm_acc = mlm_pred[mask_ids].eq(mlm_labels[mask_ids]).sum() / mask_ids.sum()
            dec_mlm_acc = dec_mlm_pred[dec_mask].eq(mlm_labels[dec_mask]).sum() / dec_mask.sum()
            sop_acc = sop_pred.eq(sop_labels).sum() / len(sop_labels)

            mlm_acc_sum += mlm_acc.item()
            dec_mlm_acc_sum += dec_mlm_acc.item()
            sop_acc_sum += sop_acc.item()

        metrics = {
            f"{metric_key_prefix}_loss": loss_sum / steps,
            f"{metric_key_prefix}_mlm_loss": mlm_loss_sum / steps,
            f"{metric_key_prefix}_sop_loss": sop_loss_sum / steps,
            f"{metric_key_prefix}_dec_mlm_loss": dec_mlm_loss_sum / steps,
            f"{metric_key_prefix}_mlm_acc": mlm_acc_sum / steps,
            f"{metric_key_prefix}_dec_mlm_acc": dec_mlm_acc_sum / steps,
            f"{metric_key_prefix}_sop_acc": sop_acc_sum / steps,
        }

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=observed_num_examples)

    def prediction_step(
            self,
            model: LogLMForPreTraining,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = None,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    _loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                # loss = loss.mean().detach()
                losses = tuple(v.mean().detach() for k, v in outputs.items() if k in self.loss_names)
                logits = tuple(v for k, v in outputs.items() if k in self.logits_names)
            else:
                losses = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs, return_dict=True)
                logits = tuple(v for k, v in outputs.items() if k in self.logits_names)

                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return losses, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return losses, logits, labels
