import json
import sys
from os.path import join
from typing import Optional, List

from dataclasses import dataclass, field, asdict


@dataclass
class _Args:
    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConstructArgs(_Args):
    model_type: str = field(metadata={"help": "Pretrained model path"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path"})
    init_model: Optional[int] = field(default=0, metadata={"choices": [0, 1], "help": "Init models' parameters"})

    noise_lambda: Optional[float] = field(default=0, metadata={"help": "NoisyTune args, generally set to 0.1 ~ 0.15"})

    layer_wise_lr_decay: Optional[float] = field(default=1.0, metadata={"help": "Layer-wise lr decay rate"})
    focal_loss_gamma: Optional[float] = field(default=0, metadata={"help": "Gamma of Focal Loss"})

    anchored_cls_length: Optional[int] = field(default=-1, metadata={"help": "Anchored CLS length (used in LogLM)"})
    pooling_all: Optional[bool] = field(default=False, metadata={"help": "Pooling all hiddens (used in LogLM)"})


@dataclass
class CBLUEDataArgs(_Args):
    cblue_root: str = field(metadata={"help": "CBLUE data root"})
    max_length: Optional[int] = field(default=128, metadata={"help": "Max sequence length"})

    # Arguments for K-fold cross validation
    curr_fold: Optional[int] = field(default=0, metadata={"help": "K-fold cross validation: current fold"})
    total_folds: Optional[int] = field(default=0, metadata={"help": "K-fold cross validation: total folds"})

    extra_train_datas: Optional[List[str]] = field(default=None, metadata={"help": "Extra training data"})

    use_rdrop: Optional[bool] = field(default=0, metadata={"help": "Use R-drop Trainer or not"})


@dataclass
class CDNDataArgs(CBLUEDataArgs):
    positive: Optional[int] = field(default=10, metadata={"help": "Positive samples"})
    negative: Optional[int] = field(default=60, metadata={"help": "Negative samples"})

    split_part: Optional[int] = field(default=-1, metadata={"help": "Split to N parts"})
    curr_part: Optional[int] = field(default=-1, metadata={"help": "Split to N parts"})

    @property
    def data_path(self):
        return join(self.cblue_root, "CHIP-CDN")

    @property
    def train_json(self):
        return join(self.data_path, "CHIP-CDN_train.json")

    @property
    def dev_json(self):
        return join(self.data_path, "CHIP-CDN_dev.json")

    @property
    def test_json(self):
        return join(self.data_path, "CHIP-CDN_test.json")

    @property
    def recall_train(self):
        return join(self.data_path, "recall_train.json")

    @property
    def recall_dev(self):
        return join(self.data_path, "recall_dev.json")

    @property
    def recall_test(self):
        return join(self.data_path, "recall_test.json")


@dataclass
class IMCSDataArgs(CBLUEDataArgs):
    ir_pred_file: Optional[str] = field(default=None, metadata={"help": "IMCS-IR test prediction file (Used in MRG)"})


@dataclass
class MDCFNPCArgs(CBLUEDataArgs):
    left_windows: Optional[int] = field(default=0, metadata={"help": "Left context length"})
    right_windows: Optional[int] = field(default=0, metadata={"help": "Right context length"})
    focus_key_ent: Optional[bool] = field(default=False, metadata={"help": "Train only entities of key sentences"})


@dataclass
class IEDataArgs(CBLUEDataArgs):
    pass


@dataclass
class ClassifyDataArgs(CBLUEDataArgs):
    task_name: Optional[str] = field(default=None, metadata={"help": "Task name"})

    @property
    def test_output_file(self):
        if self.task_name is None:
            print("[Error] Please specific the `task_name`!!!", flush=True, file=sys.stderr)
            exit(-1)
        return f"{self.task_name}_test.json"


@dataclass
class MedDGDataArgs(CBLUEDataArgs):
    n_positive: Optional[int] = field(default=1, metadata={"help": "Repeat times of positive samples"})
    n_negative: Optional[int] = field(default=30, metadata={"help": "Number of negative samples"})
