"""
Author: Da Ma
Date: 2022.01.07
Description: some trainer objects used in training. Refer to https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
"""

from transformers import Trainer, set_seed, __version__
from transformers.trainer_utils import (
    get_last_checkpoint,
    TrainOutput,
)
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.trainer_callback import TrainerState
from transformers.configuration_utils import PretrainedConfig
from typing import List, Dict, Optional, Union, Any, TYPE_CHECKING, Tuple
import os
import torch
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from utils.logger import is_main_process, get_logger

if TYPE_CHECKING:
    import optuna

TRAINER_STATE_NAME = "trainer_state.json"


def moveToCuda(data: Dict) -> Dict:
    for key in data:
        data[key] = data[key].cuda()
    return data


class PlmTrainer(Trainer):
    """
    NOTE: support update dataset after one epoch.
    """

    def __init__(
        self,
        datasetType=None,
        epochDatasetConfigList: List[Dict] = None,
        logPath: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.epochDatasetConfigList = epochDatasetConfigList
        self.epochDataIndex = 0
        self.datasetType = datasetType
        self.CE = torch.nn.CrossEntropyLoss()
        self.logger = get_logger(__name__, logPath, self.args.local_rank, False)
        self.epochTrainSteps = []

        # [FIXME: INTRODUCE MORE LOSSES HERE!]
        self.taskLosses = {}
        if "MLM" in self.model.config.tasks:
            self.taskLosses["MLMLoss"] = 0.0
            self.taskLosses["SOPLoss"] = 0.0

        if "DN" in self.model.config.tasks:
            self.taskLosses["DNLoss"] = 0.0

    def _getEpochTrainingSteps(self):
        trainingSteps = []
        for epochDatasetConfig in self.epochDatasetConfigList:
            self.train_dataset = self.datasetType(**epochDatasetConfig)
            train_dataloader = self.get_train_dataloader()
            trainingSteps.append(
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )
        self.epochTrainSteps = trainingSteps

    def _getRunnedEpochNum(self, trainingSteps: int) -> float:
        runnedEpochNum = 0
        totalEpochNum = len(self.epochDatasetConfigList)
        totalTrainingSteps = sum(self.epochTrainSteps)

        runnedEpochNum += (trainingSteps // totalTrainingSteps) * totalEpochNum
        trainingSteps %= totalTrainingSteps

        epochIndex = 0

        while trainingSteps >= self.epochTrainSteps[epochIndex]:
            runnedEpochNum += 1
            trainingSteps -= self.epochTrainSteps[epochIndex]
            epochIndex += 1

        return round(
            runnedEpochNum + trainingSteps / self.epochTrainSteps[epochIndex], 2
        )

    def _getRunnedTrainingSteps(self, runnedEpoch: float) -> int:
        totalEpochNum = len(self.epochDatasetConfig)
        totalSteps = sum(self.epochSizes)

        steps = (runnedEpoch // totalEpochNum) * totalSteps
        runnedEpoch %= totalEpochNum

        steps += sum(self.epochSizes[: int(runnedEpoch)])
        steps += math.ceil(
            self.epochSizes[int(runnedEpoch)] * (runnedEpoch - int(runnedEpoch))
        )

        return steps

    def _getStepsTrainedInCurrentEpoch(self, trainingSteps: int) -> int:
        totalTrainingSteps = sum(self.epochTrainSteps)
        trainingSteps %= totalTrainingSteps

        epochIndex = 0

        while trainingSteps >= self.epochTrainSteps[epochIndex]:
            trainingSteps -= self.epochTrainSteps[epochIndex]
            epochIndex += 1

        return trainingSteps

    def _updateTrainDataset(self, skipNum=None):
        # get the dataset of current epoch first.
        self.train_dataset = self.datasetType(
            **self.epochDatasetConfigList[self.epochDataIndex]
        )

        if skipNum is not None:
            self.train_dataset.setSkipNum(skipNum)

        # next epoch data.
        # If all data is used up, select from the beginning again.
        self.epochDataIndex = (self.epochDataIndex + 1) % len(
            self.epochDatasetConfigList
        )

    def _getUpdatedTrainDataLoader(self, skipNum=None):
        self._updateTrainDataset(skipNum=skipNum)
        return self.get_train_dataloader()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        NOTE: Most codes are from the official huggingface storehouse.
        """
        resume_from_checkpoint = (
            None if not resume_from_checkpoint else resume_from_checkpoint
        )

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:

        # ************************* PREPARATION ************************* #
        # ******** CALCULATE STEPS ******** #
        # calculate epoch training steps first.
        self.logger.info("Calculating the training steps of each epoch data...")
        self._getEpochTrainingSteps()

        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        # NOTE: Each epoch has different data sizes due to the randomness.
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = self._getRunnedEpochNum(max_steps)
            # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = self._getRunnedTrainingSteps(args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
            num_train_samples = max_steps * total_train_batch_size

        # ******** MODEL INITIALIZATION ******** #
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
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(
                    f"Can't find a valid checkpoint at {resume_from_checkpoint}"
                )

            self.logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(
                    os.path.join(resume_from_checkpoint, CONFIG_NAME)
                )
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    self.logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(
                os.path.join(resume_from_checkpoint, WEIGHTS_NAME),
                map_location="cpu",
            )
            # If the model is on the GPU, it still works!
            # self._load_state_dict_in_model(state_dict)
            self.model.load_state_dict(state_dict)

            # release memory
            del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        self.state = TrainerState()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # ******* OPTIMIZER ******* #
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # ******* PREPARE SKIPPING STEPS ******* #
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # calculate steps to skip
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

            epochs_trained = int(self._getRunnedEpochNum(self.state.global_step))
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self._getStepsTrainedInCurrentEpoch(
                    self.state.global_step
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            self.logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            self.logger.info(f"  Continuing training from epoch {epochs_trained}")
            self.logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                self.logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

        # ***************** PREPARE DATASET ***************** #
        # initialize the first epoch
        self.epochDataIndex = 0

        # ****** SKIP ****** #
        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                train_dataloader = self._getUpdatedTrainDataLoader()
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        train_dataloader = self._getUpdatedTrainDataLoader(
            steps_trained_in_current_epoch
            * args.train_batch_size
            // args.dataloader_num_workers
            - args.train_batch_size
        )

        # ************************* TRAIN ************************* #
        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {num_train_samples}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        self.logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        self.logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
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

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        for epoch in range(epochs_trained, int(num_train_epochs) + 1):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator)
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # print(epoch_iterator.dataset.skipNum)
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
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if (
                        args.max_grad_norm is not None
                        and args.max_grad_norm > 0
                        and not self.deepspeed
                    ):
                        # deepspeed does its own clipping

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, self.state.epoch
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # ** update dataloader ** #
            train_dataloader = self._getUpdatedTrainDataLoader()

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        self.logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
                dist.barrier()

            self.logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(
                self.state.best_model_checkpoint, WEIGHTS_NAME
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                self.logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        self.store_flos()

        self.is_in_train = False

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, {})

    def CELoss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        return self.CE(outputs["logits"], labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        criteria = (
            self.label_smoother if self.label_smoother is not None else self.CELoss
        )

        # ** VARIOUS LABELS ** #
        MLMLabels, SOPLabels, DNLabels = None, None, None
        # MLM Label
        if "tokenLabels" in inputs:
            MLMLabels = inputs.pop("tokenLabels")
        # SOP Label
        if "sentLabels" in inputs:
            SOPLabels = inputs.pop("sentLabels")
        # DN Label
        if "dnLabels" in inputs:
            DNLabels = inputs.pop("dnLabels")

        # [FIXME: INTRODUCE MORE LABELS HERE!]

        outputs = model(**inputs)

        # ** VARIOUS LOSSES ** #
        loss = 0

        def _reshape(tensor):
            return tensor.view(-1, tensor.size(-1))

        # [FIXME: INTRODUCE MORE LOSSES HERE!]

        lossDict = {}
        # MLM loss
        if MLMLabels is not None:
            MLMLoss = self.CELoss({"logits": _reshape(outputs[0])}, MLMLabels.view(-1))
            loss += MLMLoss
            lossDict["MLMLoss"] = MLMLoss

        # SOP loss
        if SOPLabels is not None:
            SOPLoss = criteria({"logits": _reshape(outputs[1])}, SOPLabels.view(-1))
            loss += SOPLoss
            lossDict["SOPLoss"] = SOPLoss

        # DN loss
        if DNLabels is not None:
            DNLoss = self.CELoss({"logits": _reshape(outputs[2])}, DNLabels.view(-1))
            loss += DNLoss
            lossDict["DNLoss"] = DNLoss

        lossDict["totalLoss"] = loss
        if return_outputs:
            lossDict["outputs"] = outputs

        return lossDict

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        lossDict = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            # mean() to average on multi-gpu parallel training
            for key in lossDict:
                lossDict[key] = lossDict[key].mean()

        if self.args.gradient_accumulation_steps > 1:
            for key in lossDict:
                lossDict[key] = lossDict[key] / self.args.gradient_accumulation_steps

        for key in self.taskLosses:
            self.taskLosses[key] += lossDict[key].detach()

        totalLoss = lossDict["totalLoss"]
        totalLoss.backward()

        return totalLoss.detach()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            deltaSteps = self.state.global_step - self._globalstep_last_logged
            logs["totalLoss"] = round(
                tr_loss_scalar / deltaSteps,
                4,
            )

            for key in self.taskLosses:
                loss_scalar = self.taskLosses[key].item()
                self.taskLosses[key] -= self.taskLosses[key]
                logs[key] = round(loss_scalar / deltaSteps, 4)

            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate and is_main_process(self.args.local_rank):
            metrics = self.evaluate()
            msg = f"\n\n======= EVALUATION REPORT =======\n>>> steps: {self.state.global_step}\n"
            for metric in metrics:
                msg += f">>> {metric}: {metrics[metric]}\n"
            msg += "=================================\n"
            self.logger.info(msg)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
            self,
            eval_dataset: Optional[torch.utils.data.Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        batch_size = eval_dataloader.batch_size
        num_examples = self.num_examples(eval_dataloader)
        self.logger.info("***** Running Evaluation *****")
        self.logger.info(f"  Num examples = {num_examples}")
        self.logger.info(f"  Batch size = {batch_size}")

        with torch.no_grad():
            predictions, labels = self.prediction_loop(
                eval_dataloader, description="Evaluation", ignore_keys=ignore_keys
            )

        metrics = {}

        # [FIXME: INTRODUCE MORE EVALUATION HERE!]

        if "MLM" in self.model.config.tasks:
            # MLM metric
            MLMPreds = predictions[0]
            MLMLabels = labels[0]
            # print(MLMPreds[0])
            # print(MLMLabels[0])
            mask = torch.ne(MLMLabels, -100)
            accMatrix = torch.logical_and(torch.eq(MLMPreds, MLMLabels), mask)
            MLMACC = torch.sum(accMatrix).item() / torch.sum(mask).item()
            metrics["MLMACC"] = round(MLMACC, 4)

            # SOP metric
            SOPPreds = predictions[1]
            SOPLabels = labels[1]
            NUM = SOPLabels.size(0) if SOPLabels.dim() > 0 else 1
            SOPACC = torch.sum(torch.eq(SOPPreds, SOPLabels)).item() / NUM
            metrics["SOPACC"] = round(SOPACC, 4)

        if "DN" in self.model.config.tasks:
            # DNACC
            # given the ground truth history, calculate the accuracy of next token
            DNPreds = predictions[2]
            DNLabels = labels[2]

            mask = torch.ne(DNLabels, -100)
            accMatrix = torch.logical_and(torch.eq(DNPreds, DNLabels), mask)
            DNACC = torch.sum(accMatrix).item() / torch.sum(mask).item()
            metrics["DNACC"] = round(DNACC, 4)

        return metrics

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = None,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # pop labels
        # [FIXME: POP LABELS HERE!]
        inputs = moveToCuda(inputs)

        labels = tuple()

        if "MLM" in self.model.config.tasks:
            tokenLabels = inputs.pop("tokenLabels")
            sentLabels = inputs.pop("sentLabels")
            labels = labels + (tokenLabels, sentLabels)

        if "DN" in self.model.config.tasks:
            DNLabels = inputs.pop("dnLabels")
            labels = labels + (DNLabels,)

        outputs = model(**inputs)
        if isinstance(outputs, dict):
            preds = tuple(
                torch.argmax(v, dim=-1, keepdim=True)
                for k, v in outputs.items()
                if k not in ignore_keys
            )
        else:
            preds = tuple(torch.argmax(v, dim=-1, keepdim=True) for v in outputs)

        return preds, labels

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Tuple[torch.tensor, torch.tensor]:
        predsList, labelsList = [], []

        self.model.eval()

        for batch in tqdm(dataloader):
            preds, labels = self.prediction_step(self.model, batch)
            predsList.append(preds)
            labelsList.append(labels)

        def catTuple(data: List[Tuple[torch.tensor]]) -> List[torch.tensor]:
            if not data:
                return data
            catData = []
            num = len(data[0])

            for i in range(num):
                catData.append(
                    torch.cat(list(map(lambda x: x[i], data)), dim=0).squeeze()
                )
            return catData

        return catTuple(predsList), catTuple(labelsList)
