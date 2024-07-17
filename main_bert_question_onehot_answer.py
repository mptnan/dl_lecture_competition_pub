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
    VQABertEmbeddingModel,
    VQAStrQuestionOneHotAnswerDataset,
    get_all_sentences,
    get_yes_no,
    preprocess,
    process_text,
)


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    timer=None,
    n_top=10,
    n_bottom=10,
):
    model.train()

    total_loss = 0
    total_acc = 0

    start = time.time()
    if timer is not None:
        timer.push()

    preds = []

    for (
        image,
        question,
        answer_tensor,
        answers,
        idx,
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
        preds.extend(list(pred.argmax(1)))

    return total_loss / len(dataloader), total_acc / len(dataloader), time.time() - start, preds


class gcn:
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean) / (std + 10 ** (-6))


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
            transforms.RandomRotation((-180, 180)),
            gcn(),
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

    with open("aidx.txt", "w") as f:
        for i in aidx.idx_to_str:
            f.write(f"{i}\n")

    epoch_timer.push()
    logger.info(f"preparation took {epoch_timer.last_lap()/60:.2f} minutes")

    # train model
    # 10 mins of TPU / epoch
    for epoch in range(num_epoch):
        train_loss, train_acc, train_time, preds = train(
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
        image = image.to(device)
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
    main()
