import multiprocessing
import random
import re
import time
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from statistics import mode

import hydra
import numpy as np
import pandas
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms


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


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # 冠詞の削除
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't",
        "isnt": "isn't",
        "arent": "aren't",
        "wont": "won't",
        "cant": "can't",
        "wouldnt": "wouldn't",
        "couldnt": "couldn't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", " ", text)

    # 句読点をスペースに変換
    text = re.sub(r"\s+,", ",", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換
            except KeyError:
                question[-1] = 1  # 未知語

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)


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


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, n_answer))

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device, timer=None):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    if timer is not None:
        timer.push()
    for image, question, answers, mode_answer in dataloader:
        if timer is not None:
            timer.push(tag="load_data")

        image, question, answers, mode_answer = (
            image.to(device),
            question.to(device),
            answers.to(device),
            mode_answer.to(device),
        )
        if timer is not None:
            timer.push(tag="to_device")

        pred = model(image, question)
        if timer is not None:
            timer.push(tag="pred")

        loss = criterion(pred, mode_answer.squeeze())
        if timer is not None:
            timer.push(tag="calc_loss")

        optimizer.zero_grad()
        loss.backward()
        if timer is not None:
            timer.push(tag="backward")

        optimizer.step()
        if timer is not None:
            timer.push(tag="step")

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        if timer is not None:
            timer.push()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = (
            image.to(device),
            question.to(device),
            answers.to(device),
            mode_answer.to(device),
        )

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


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


class DeviceNotAvailable(RuntimeError):
    pass


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger = prepare_logger()
    logger.info("[configuration]\n" + OmegaConf.to_yaml(cfg))

    # redefine all cfg variables
    seed = cfg.env.seed
    num_epoch = cfg.env.num_epoch
    lr = cfg.env.lr
    num_workers = cfg.env.num_workers
    device = cfg.env.device

    # deviceの設定
    logger.info(f"{str(device)} is used for device")

    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"output into {hydra_output_dir}")

    logger.info(f"{str(device)} is used for device")

    devices = device_info()
    logger.info("[devices]\n" + devices["info"])
    if device == "xla" and not devices["tpu"]["available"]:
        raise DeviceNotAvailable("tpu(xla) is not available")
    elif device == "cuda" and not devices["cuda"]["available"]:
        raise DeviceNotAvailable("gpu(cuda) is not available")

    set_seed(seed)

    epoch_timer = Timer()
    train_timer = Timer()

    # dataloader / model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = VQADataset(
        df_path="./data/train.json",
        image_dir="./data/train",
        transform=transform,
    )
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    model = VQAModel(vocab_size=len(train_dataset.question2idx) + 1, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epoch_timer.push()
    logger.info(f"preparation took {epoch_timer.last_lap()/60:.2f} minutes")

    # train model
    # 10 mins of TPU / epoch
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            timer=None,
        )
        _msg = "\n".join(
            [
                f"epoch【{epoch + 1}/{num_epoch}】",
                f"train time: {train_time:.2f} [s]",
                f"train loss: {train_loss:.4f}",
                f"train acc: {train_acc:.4f}",
                f"train simple acc: {train_simple_acc:.4f}",
            ]
        )
        logger.info(_msg)
        epoch_timer.push()
        logger.info(f"epoch took {epoch_timer.last_lap()/60:.2f} minutes")

        if "load_data" in train_timer._tag_laps.keys():
            _msg = "\n".join(
                [
                    "[train_timer] average secs",
                    f"load_data: {train_timer.average_lap_tag('load_data'):.2e}",
                    f"to_device took {train_timer.average_lap_tag('to_device'):.2e} secs average",
                    f"pred took {train_timer.average_lap_tag('pred'):.2e} secs average",
                    f"calc_loss took {train_timer.average_lap_tag('calc_loss'):.2e} secs average",
                    f"backward took {train_timer.average_lap_tag('backward'):.2e} secs average",
                    f"step took {train_timer.average_lap_tag('step'):.2e} secs average",
                ]
            )
            logger.info(_msg)

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), hydra_output_dir / "model.pth")
    np.save(hydra_output_dir / "submission.npy", submission)


if __name__ == "__main__":
    main()
