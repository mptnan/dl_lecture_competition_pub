import multiprocessing
import os
import random
import time
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

import numpy as np
import torch
import yaml

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
