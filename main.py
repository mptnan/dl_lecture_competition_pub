import shutil
import time
from collections import Counter

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtext.data.utils import get_tokenizer
from torchvision import transforms
from tqdm import tqdm

from src import (
    CustomVocab,
    Timer,
    VQA_criterion,
    VQACorpusDataset,
    VQAEmbeddingModel,
    VQAOneHotAnswerDataset2,
    VQASampleModel,
    VQAStrQuestionOneHotAnswerDataset,
    get_all_sentences,
    get_answers,
    get_yes_no,
    preprocess,
    process_text,
)


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
    preds = []

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

        image, question, answer_tensor, answers = (
            image.to(device),
            question.to(device),
            answer_tensor.to(device),
            answers.to(device),
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
        simple_acc += (pred.argmax(1) == torch.mode(answers, dim=1)).float().mean().item
        ()  # simple accuracy

        preds.extend(pred.argmax(1).tolist())
        if timer is not None:
            timer.push()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start, preds


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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

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

    trainval_dataset = VQAOneHotAnswerDataset2(
        df_path="./data/train.json",
        image_dir="./data/train",
        transform=transform,
        answer=True,
        onehot_type="global_mode",
    )
    aidx = trainval_dataset.aidx

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

    model = VQASampleModel(
        vocab_size=len(trainval_dataset.idx2question),
        n_answer=len(trainval_dataset.aidx),
    ).to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epoch_timer.push()
    logger.info(f"preparation took {epoch_timer.last_lap()/60:.2f} minutes")

    # train model
    # 10 mins of TPU / epoch
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time, preds = train(
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
        with open("tmp_preds.txt", "w") as f:
            for i in preds:
                f.write(f"{i}\n")
        c = Counter(preds)
        _msg = "preds freq:\n" + "\n".join(
            [f"{idx}, {aidx.idx_to_str[idx]}: {freq}/{len(preds)}" for idx, freq in list(c.most_common())[:3]],
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


if __name__ == "__main__":
    main()
