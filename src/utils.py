import json
import multiprocessing
import os
import random
import time
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from .data import process_text

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # mute tensorflow warnings
import tensorflow as tf


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.0

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.0
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


class Timer:
    def __init__(self):
        self._last_time = time.perf_counter()
        self._laps = []
        self._tag_laps = {}

    def push(self, tag: str = ""):
        t = time.perf_counter()
        lap = t - self._last_time
        self._last_time = t
        self._laps.append(lap)
        if tag not in self._tag_laps.keys():
            self._tag_laps[tag] = []
        self._tag_laps[tag].append(lap)

    def last_lap(self) -> float:
        if len(self._laps) == 0:
            return 0.0
        return self._laps[-1]

    def last_lap_tag(self, tag: str):
        if tag not in self._tag_laps.keys() or len(self._tag_laps[tag]) == 0:
            return 0.0
        return self._tag_laps[tag][-1]

    def average_lap(self) -> float:
        if len(self._laps) == 0:
            return 0.0
        return sum(self._laps) / len(self._laps)

    def average_lap_tag(self, tag: str):
        if tag not in self._tag_laps.keys() or len(self._tag_laps[tag]) == 0:
            return 0.0
        return sum(self._tag_laps[tag]) / len(self._tag_laps[tag])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_logger():
    logger = getLogger("dl_lecture_main.py")
    if logger.hasHandlers():
        return logger

    logger.setLevel(DEBUG)
    h = StreamHandler()
    h.setLevel(INFO)
    f = Formatter("[%(levelname)s][%(asctime)s]: %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
    return logger


def device_info():
    devices = {}

    # tpu
    devices["tpu"] = {}
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        devices["tpu"]["available"] = True
        devices["tpu"]["num_accelerators"] = tpu.num_accelerators()["TPU"]
    except ValueError:
        devices["tpu"]["available"] = False

    # gpu cuda
    devices["cuda"] = {}
    if torch.cuda.is_available():
        devices["cuda"]["available"] = True
        devices["cuda"]["device_count"] = torch.cuda.device_count()
    else:
        devices["cuda"]["available"] = False

    # cpu
    devices["cpu"] = {}
    devices["cpu"]["available"] = True
    devices["cpu"]["cpu_count"] = multiprocessing.cpu_count()

    devices["info"] = "available devices:\n" + yaml.dump(devices)

    return devices


def get_yes_no(msg):
    while True:
        reply = str(input(msg + " (Y/n): ")).lower().strip()
        if reply == "y":
            return True
        if reply == "n":
            return False


def get_question_max_sentence_length() -> int:
    sentences = []
    for file in ["./data/train.json", "./data/valid.json"]:
        with open(file) as f:
            data = json.load(f)
        sentences.extend(data["question"].values())

    max_len = 0
    for s in sentences:
        tmp_max_len = len(process_text(s).split(" "))
        if tmp_max_len > max_len:
            max_len = tmp_max_len
            print(max_len, s)

    return max_len  # 56


def get_all_sentences() -> list[str]:
    res = []
    for p in ["./data/train.json", "./data/valid.json"]:
        df = pd.read_json(p)
        res.extend(list(df["question"]))

    return res


def get_answers() -> list[str]:
    res = []
    df = pd.read_csv("./data/class_mapping.csv")
    res.extend(list(df["answer"]))
    for p in ["./data/train.json"]:
        df = pd.read_json(p)
        for dd in df["answers"]:
            for each_answer in dd:
                res.append(each_answer["answer"])
    return list(res)


class DeviceNotAvailable(RuntimeError):
    pass


def preprocess(cfg: DictConfig):
    """
    Return:
        logger
        seed
        num_epoch
        lr
        num_workers
        device
        env_name
        ask_save_model
        default_save_model
        hydra_output_dir
    runtime_output_dir
    """

    logger = prepare_logger()
    logger.info("[configuration]\n" + OmegaConf.to_yaml(cfg))

    # redefine all cfg variables
    seed = cfg.env.seed
    num_epoch = cfg.env.num_epoch
    lr = cfg.env.lr
    num_workers = cfg.env.num_workers
    device = cfg.env.device
    env_name = cfg.env.env_name
    ask_save_model = cfg.env.ask_save_model
    default_save_model = cfg.env.default_save_model

    # deviceの設定
    logger.info(f"{str(device)} is used for device")

    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"output into {hydra_output_dir}")
    runtime_output_dir = Path(cfg.env.runtime_output_dir)
    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"runtime output into {runtime_output_dir}")

    devices = device_info()
    logger.info("[devices]\n" + devices["info"])
    if device == "xla" and not devices["tpu"]["available"]:
        raise DeviceNotAvailable("tpu(xla) is not available")
    elif device == "cuda" and not devices["cuda"]["available"]:
        raise DeviceNotAvailable("gpu(cuda) is not available")
    elif device not in ["cpu", "cuda", "xla"]:
        raise DeviceNotAvailable(f"{device} is not available")

    if device == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

    set_seed(seed)

    return (
        logger,
        seed,
        num_epoch,
        lr,
        num_workers,
        device,
        env_name,
        ask_save_model,
        default_save_model,
        hydra_output_dir,
        runtime_output_dir,
    )
