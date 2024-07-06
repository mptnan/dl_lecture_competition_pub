import multiprocessing
import os
import random
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # mute tensorflow warnings
import tensorflow as tf


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
