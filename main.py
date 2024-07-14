import json
import shutil
import time
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchtext.data.utils import get_tokenizer
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from src import (
    CustomVocab,
    ResNet18,
    ResNet50,
    Timer,
    VQA_criterion,
    VQACorpusDataset,
    VQAOneHotAnswerDataset,
    VQAStrQuestionOneHotAnswerDataset,
    device_info,
    get_yes_no,
    prepare_logger,
    process_text,
    set_seed,
)


def get_question_max_sentence_length() -> int:
    sentences = []
    for file in ["./data/train.json", "./data/valid.json"]:
        with open("./data/train.json") as f:
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


class InvalidResnetType(RuntimeError):
    pass


class VQABertEmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        resnet_type: Literal[18, 50],
        n_answer: int,
        device: str,
    ):
        super().__init__()
        if resnet_type == 18:
            self.resnet = ResNet18()
        elif resnet_type == 50:
            self.resnet = ResNet50()
        else:
            raise InvalidResnetType

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer),
            nn.Softmax(dim=1),
        )

        self.device = device

    def forward(self, image: torch.Tensor, question: torch.Tensor):
        # image: (*, C, H, W)
        # question: (*,), type=str
        # -> (*, n_answer)
        image_feature = self.resnet(image)  # (*, C, H, W)->(*, 512)

        question_input = self.bert_tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )  # (*, L)->(*, embedding_dim)
        question_input = {k: v.to(self.device) for k, v in question_input.items()}

        with torch.no_grad():
            outputs = self.bert_model(**question_input)

        question_feature = outputs.last_hidden_state[:, 0, :]  # [CLS]トークンの特徴量を使用
        # (*, 768)

        x = torch.cat([image_feature, question_feature], dim=1)  # (*, 512 + 768)
        x = self.fc(x)  # (*, 512 + 768)->(*, n_answer)

        return x


class VQAEmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        resnet_type: Literal[18, 50],
        lstm_hidden_dim: int,
        lstm_bidirectional: bool,
        n_answer: int,
    ):
        super().__init__()
        if resnet_type == 18:
            self.resnet = ResNet18()
        elif resnet_type == 50:
            self.resnet = ResNet50()
        else:
            raise InvalidResnetType

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )
        lstm_output_hidden_dim = (2 if lstm_bidirectional else 1) * lstm_hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(512 + lstm_output_hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer),
            nn.Softmax(dim=1),
        )

    def forward(self, image: torch.Tensor, question: torch.Tensor):
        # image: (*, C, H, W)
        # question: (*, L)  L: length of a sentence
        # -> (*, n_answer)
        image_feature = self.resnet(image)  # (*, C, H, W)->(*, 512)

        question = self.embed(question)  # (*, L)->(*, embedding_dim)
        _, (h, _) = self.lstm(question)  # (*, embedding_dim)->(n_direction, *, lstm_hidden_dim)
        if self.lstm_bidirectional:
            question_feature = torch.cat([h[0], h[1]], dim=1)  # (*, lstm_output_hidden_dim)
        else:
            question_feature = h[0]

        x = torch.cat([image_feature, question_feature], dim=1)  # (*, 512 + lstm_output_hidden_dim)
        x = self.fc(x)  # (*, 512 + lstm_output_hidden_dim)->(*, n_answer)

        return x


# 4. 学習の実装
def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    timer=None,
):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    if timer is not None:
        timer.push()
    for image, question, answers, mode_answer in tqdm(
        dataloader,
        total=len(dataloader),
        leave=False,
    ):
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


def train_onehot_answer(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    timer=None,
):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    if timer is not None:
        timer.push()
    for (
        image,
        question,
        answer_tensor,
        answers,
    ) in tqdm(
        dataloader,
        total=len(dataloader),
        leave=False,
    ):
        if timer is not None:
            timer.push(tag="load_data")

        image, question, answer_tensor = (
            image.to(device),
            question.to(device),
            answer_tensor.to(device),
        )
        if timer is not None:
            timer.push(tag="to_device")

        pred = model(image, question)
        if timer is not None:
            timer.push(tag="pred")

        loss = criterion(pred, answer_tensor)
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
        if timer is not None:
            timer.push()

    return total_loss / len(dataloader), total_acc / len(dataloader), time.time() - start


def train_bert_question_onehot_answer(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    timer=None,
):
    model.train()

    total_loss = 0
    total_acc = 0

    start = time.time()
    if timer is not None:
        timer.push()
    for (
        image,
        question,
        answer_tensor,
        answers,
    ) in tqdm(
        dataloader,
        total=len(dataloader),
        leave=False,
    ):
        if timer is not None:
            timer.push(tag="load_data")

        image, answer_tensor = (
            image.to(device),
            answer_tensor.to(device),
        )
        if timer is not None:
            timer.push(tag="to_device")

        pred = model(image, question)
        if timer is not None:
            timer.push(tag="pred")

        loss = criterion(pred, answer_tensor)
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
        if timer is not None:
            timer.push()

    return total_loss / len(dataloader), total_acc / len(dataloader), time.time() - start


def eval(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(
        dataloader,
        total=len(dataloader),
        leave=False,
    ):
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


class DeviceNotAvailable(RuntimeError):
    pass


def preprocess(cfg: DictConfig):
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_legacy(cfg: DictConfig):

    (
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
    ) = preprocess(cfg)

    epoch_timer = Timer()
    train_timer = Timer()

    # dataloader / model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # make vocab
    all_sentences = get_all_sentences()
    vocab = CustomVocab(text_processor=process_text, tokenizer=get_tokenizer("basic_english"))
    for s in all_sentences:
        vocab.add_sentence(s)
    vocab.set_vocab(min_freq=25)

    # make answers list
    answers = get_answers()
    answer_vocab = CustomVocab(text_processor=process_text, tokenizer=lambda x: [x])
    for a in answers:
        answer_vocab.add_sentence(a)
    answer_vocab.set_vocab(min_freq=1)

    trainval_dataset = VQACorpusDataset(
        df_path="./data/train.json",
        image_dir="./data/train",
        len_sentence=256,
        vocab=vocab,
        transform=transform,
        answer=True,
        answer_vocab=answer_vocab,
    )

    test_dataset = VQACorpusDataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        len_sentence=256,
        vocab=vocab,
        transform=transform,
        answer=False,
    )

    # train_size = len(trainval_dataset) * 0.8
    # val_size = len(trainval_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=128,
    #     shuffle=True,
    #     num_workers=num_workers,
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=len(val_dataset),
    #     shuffle=False,
    #     num_workers=num_workers,
    # )

    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
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

    model = VQAEmbeddingModel(
        vocab_size=len(vocab),
        resnet_type=18,
        embedding_dim=512,
        n_answer=len(answer_vocab),
        lstm_bidirectional=False,
        lstm_hidden_dim=512,
    ).to(device)

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

    submission = [answer_vocab.itow(id) for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), runtime_output_dir / "model.pth")
    np.save(hydra_output_dir / "submission.npy", submission)

    if runtime_output_dir.resolve() != hydra_output_dir.resolve():
        if ask_save_model:
            if get_yes_no("Do you save the trained model?"):
                shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")
        elif default_save_model:
            shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_onehot_answer(cfg: DictConfig):
    (
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
    ) = preprocess(cfg)

    epoch_timer = Timer()
    train_timer = Timer()

    # dataloader / model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # make vocab
    all_sentences = get_all_sentences()
    vocab = CustomVocab(text_processor=process_text, tokenizer=get_tokenizer("basic_english"))
    for s in all_sentences:
        vocab.add_sentence(s)
    vocab.set_vocab(min_freq=25)

    trainval_dataset = VQAOneHotAnswerDataset(
        df_path="./data/train.json",
        image_dir="./data/train",
        len_sentence=256,
        vocab=vocab,
        transform=transform,
        answer=True,
        onehot_type="most_confident_mode",
    )

    test_dataset = VQACorpusDataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        len_sentence=256,
        vocab=vocab,
        transform=transform,
        answer=False,
    )

    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
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
    aidx = trainval_dataset.aidx
    model = VQAEmbeddingModel(
        vocab_size=len(vocab),
        resnet_type=50,
        embedding_dim=512,
        n_answer=len(aidx),
        lstm_bidirectional=True,
        lstm_hidden_dim=512,
    ).to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epoch_timer.push()
    logger.info(f"preparation took {epoch_timer.last_lap()/60:.2f} minutes")

    # train model
    # 10 mins of TPU / epoch
    for epoch in range(num_epoch):
        train_loss, train_acc, train_time = train_onehot_answer(
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

    submission = [aidx.idx_to_str[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), runtime_output_dir / "model.pth")
    np.save(hydra_output_dir / "submission.npy", submission)

    if runtime_output_dir.resolve() != hydra_output_dir.resolve():
        if ask_save_model:
            if get_yes_no("Do you save the trained model?"):
                shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")
        elif default_save_model:
            shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_bert_question_onehot_answer(cfg: DictConfig):
    (
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
    ) = preprocess(cfg)

    epoch_timer = Timer()
    train_timer = Timer()

    # dataloader / model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # make vocab
    all_sentences = get_all_sentences()
    vocab = CustomVocab(text_processor=process_text, tokenizer=get_tokenizer("basic_english"))
    for s in all_sentences:
        vocab.add_sentence(s)
    vocab.set_vocab(min_freq=25)

    trainval_dataset = VQAStrQuestionOneHotAnswerDataset(
        df_path="./data/train.json",
        image_dir="./data/train",
        transform=transform,
        answer=True,
        onehot_type="most_confident_mode",
    )

    test_dataset = VQAStrQuestionOneHotAnswerDataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        transform=transform,
        answer=False,
    )

    train_loader = torch.utils.data.DataLoader(
        trainval_dataset,
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
    aidx = trainval_dataset.aidx
    model = VQABertEmbeddingModel(
        vocab_size=len(vocab),
        resnet_type=50,
        embedding_dim=512,
        n_answer=len(aidx),
        device=device,
    ).to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epoch_timer.push()
    logger.info(f"preparation took {epoch_timer.last_lap()/60:.2f} minutes")

    # train model
    # 10 mins of TPU / epoch
    for epoch in range(num_epoch):
        train_loss, train_acc, train_time = train_bert_question_onehot_answer(
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

    submission = [aidx.idx_to_str[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), runtime_output_dir / "model.pth")
    np.save(hydra_output_dir / "submission.npy", submission)

    if runtime_output_dir.resolve() != hydra_output_dir.resolve():
        if ask_save_model:
            if get_yes_no("Do you save the trained model?"):
                shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")
        elif default_save_model:
            shutil.copyfile(runtime_output_dir / "model.pth", hydra_output_dir / "model.pth")


if __name__ == "__main__":
    main_bert_question_onehot_answer()
