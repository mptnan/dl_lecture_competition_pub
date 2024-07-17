import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mode, multimode
from typing import Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler
from torchtext.vocab import Vocab, vocab
from torchvision import transforms


def process_text(text: str) -> str:
    """
    sentence: str -> processed sentence: str
    """
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

    # 半角英数字、空白文字、'、:以外をスペースに変換（マルチバイト文字は？）
    text = re.sub(r"[^\w\s':]", " ", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


class CustomVocab:
    """ """

    def __init__(self, text_processor: Callable[[str], str], tokenizer: Callable[[str], list[str]]):
        self._counter = Counter()
        self._text_processor = text_processor
        self._tokenizer = tokenizer
        self.vocab = None

    def add_sentence(self, sentence: str) -> None:
        self._counter.update(self._tokenizer(self._text_processor(sentence)))

    def set_vocab(self, min_freq: int):
        self.vocab = vocab(
            self._counter,
            min_freq=min_freq,
            specials=["<unk>", "<PAD>", "<BOS>", "<EOS>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

    def to_tensor(self, sentence: str, lsize: int) -> torch.Tensor:
        sentence = self._text_processor(sentence)
        text = [self.vocab[token] for token in self._tokenizer(sentence)][: lsize - 2]
        text = [self.vocab["<BOS>"]] + text + [self.vocab["<EOS>"]]
        pad = [self.vocab["<PAD>"]] * (lsize - len(text))

        return torch.tensor(text + pad, dtype=torch.int)

    def itow(self, idx: int):
        return self.vocab.get_itos()[idx]

    def wtoi(self, word: str):
        return self.vocab[self._text_processor(word)]

    def __len__(self):
        return len(self.vocab)


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        answer: bool = True,
    ):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
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

    def update_dict(self, dataset: torch.utils.data.Dataset):
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

    def __getitem__(self, idx: int) -> Union[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
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
        question_words = process_text(self.df["question"][idx]).split(" ")

        # question: idxの質問文に含まれる単語に1それ以外に0の入ったベクトル
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1
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


class VQACorpusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str,
        image_dir: str,
        len_sentence: int,
        vocab: CustomVocab,
        transform: Optional[transforms.Compose] = None,
        answer: bool = True,
        answer_vocab: Optional[CustomVocab] = None,
    ):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame

        self.questions = []  # (n_questions, len_sentence)
        for question in self.df["question"]:
            words = vocab.to_tensor(question, len_sentence)  # (len_sentence,)
            self.questions.append(words)

        self.answer = answer
        if self.answer:
            self.answer_vocab = answer_vocab

    def __getitem__(self, idx: int) -> Union[
        tuple[
            torch.Tensor,
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[
            torch.Tensor,
            list[torch.Tensor],
        ],
    ]:
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
        question : torch.Tensor  (vocab_size, n_words_in_sentence)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        if self.answer:
            answers = [self.answer_vocab.wtoi(answer["answer"]) for answer in self.df["answers"][idx]]  # :list[int]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, self.questions[idx], torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, self.questions[idx]

    def __len__(self):
        return len(self.df)


def process_answer(text: str) -> str:
    """
    配布されたprocess_textに同じ
    answerの前処理として常に使う
    """
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # 冠詞の削除
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # 短縮形のカンマの追加
    contractions = {"dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't", "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"}
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", " ", text)

    # 句読点をスペースに変換
    text = re.sub(r"\s+,", ",", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


@dataclass
class AnswerIndex:
    idx_to_str: Mapping[int, str]
    str_to_idx: Mapping[str, int]

    def __len__(self) -> int:
        return len(self.idx_to_str)


def all_answers_list(train_json_path: str) -> AnswerIndex:
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res = []
    for ans_l in answers.values():
        for ans in ans_l:
            res.append(process_answer(ans["answer"]))

    idx_to_str: list[str] = list(set(res))  # unique
    str_to_idx: Mapping[str, int] = {v: i for i, v in enumerate(idx_to_str)}
    return AnswerIndex(
        idx_to_str=idx_to_str,
        str_to_idx=str_to_idx,
    )


def answer_indices_to_tensor(ints: list[int], max_size: int) -> torch.Tensor:
    """
    [0, 2, 3] -> [0.333, 0, 0.333, 0.333, ...]
    """
    res = torch.zeros(max_size)
    for i in ints:
        res[i] += 1 / len(ints)

    return res / torch.sum(res)


def global_mode_tensors(train_json_path: str, aidx: AnswerIndex) -> Mapping[str, torch.Tensor]:
    # 10個の回答のうちの最頻値
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res = {}
    for k, ans_l in answers.items():
        tmp = [aidx.str_to_idx[process_answer(ans["answer"])] for ans in ans_l]
        res[k] = answer_indices_to_tensor(multimode(tmp), len(aidx))

    return res


def get_most_confident(confidences: list[str]) -> str:
    if "yes" in confidences:
        return "yes"
    elif "maybe" in confidences:
        return "maybe"
    return "no"


def most_confident_mode_tensors(train_json_path: str, aidx: AnswerIndex) -> Mapping[str, torch.Tensor]:
    # 10個の回答のうち最も信頼性の高い回答のうちの最頻値
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res = {}
    for k, ans_l in answers.items():
        cf = get_most_confident([ans["answer_confidence"] for ans in ans_l])
        tmp = [aidx.str_to_idx[process_answer(ans["answer"])] for ans in ans_l if ans["answer_confidence"] == cf]
        res[k] = answer_indices_to_tensor(multimode(tmp), len(aidx))

    return res


def global_mode_except_unanswerable_tensors(train_json_path: str, aidx: AnswerIndex) -> Mapping[str, torch.Tensor]:
    # 10個の回答のうち最も信頼性の高い回答のうちの最頻値
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res = {}
    for k, ans_l in answers.items():
        tmp = [aidx.str_to_idx[process_answer(ans["answer"])] for ans in ans_l if process_answer(ans["answer"]) != "unanswerable"]
        if not tmp:
            tmp = [aidx.str_to_idx["unanswerable"]]
        res[k] = answer_indices_to_tensor(multimode(tmp), len(aidx))

    return res


def one_hot_vector_tensor(idx: int, max_size: int) -> torch.Tensor:
    t = torch.zeros(max_size)
    t[idx] = 1
    return t


def confidence_weighted_average_tensors(
    train_json_path: str,
    aidx: AnswerIndex,
    confidence_weight: Mapping[str, float] = {"yes": 1},
) -> Mapping[str, torch.Tensor]:
    # 10個の回答のうち最も信頼性の高い回答のうちの最頻値
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    weights = {}
    res = {}
    for k, ans_l in answers.items():
        for ans in ans_l:
            conf = ans["answer_confidence"]
            word = ans["answer"]
            if conf in confidence_weight.keys():
                ans_idx = process_answer(word)
                if ans_idx not in weights.keys():
                    weights[ans_idx] = 0
                weights[ans_idx] += confidence_weight[conf]
        res_t = torch.zeros(len(aidx))
        for wk, wv in weights.items():
            res_t += wv * one_hot_vector_tensor(wk, len(aidx))

        res[k] = res_t / torch.sum(res_t)

    return res


def get_answers(train_json_path: str, aidx: AnswerIndex) -> Mapping[str, list[int]]:
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res: Mapping[str, list[int]] = {}
    for k, ans_l in answers.items():
        res[k] = [aidx.str_to_idx[process_answer(ans["answer"])] for ans in ans_l]

    return res


def get_answers_tensor(train_json_path: str, aidx: AnswerIndex) -> torch.Tensor:  # (n_questions, n_answer_vocab)
    with open(train_json_path) as f:
        data = json.load(f)
    answers = data["answers"]
    res: Mapping[str, list[int]] = torch.full((len(answers), len(answers["0"])), 0.0)
    for k, ans_l in answers.items():
        res[int(k)] = torch.tensor([aidx.str_to_idx[process_answer(ans["answer"])] for ans in ans_l])

    return res


class NoModeTypeError(RuntimeError):
    pass


class VQAOneHotAnswerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str,
        image_dir: str,
        len_sentence: int,
        vocab: CustomVocab,
        transform: Optional[transforms.Compose] = None,
        answer: bool = True,
        onehot_type: Optional[
            Literal[
                "global_mode",
                "most_confident_mode",
                "confidence_weighted_average",
            ]
        ] = None,
        confidence_weight: Optional[Mapping[str, float]] = None,
    ):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        with open(df_path) as f:
            self.json = json.load(f)  # 画像ファイルのパス，question, answerを持つDataFrame

        self.questions = {}  # (n_questions, len_sentence)
        for k, question in self.json["question"].items():
            words = vocab.to_tensor(question, len_sentence)  # (len_sentence,)
            self.questions[k] = words

        self.answer = answer
        if self.answer:
            self.aidx = all_answers_list(df_path)
            if onehot_type == "global_mode":
                self.answer_tensors = global_mode_tensors(df_path, self.aidx)
            elif onehot_type == "most_confident_mode":
                self.answer_tensors = most_confident_mode_tensors(df_path, self.aidx)
            elif onehot_type == "confidence_weighted_average":
                self.answer_tensors = confidence_weighted_average_tensors(
                    df_path,
                    self.aidx,
                    confidence_weight=confidence_weight,
                )
            else:
                raise NoModeTypeError

            self.answers = get_answers(df_path, self.aidx)

    def __getitem__(self, idx: int) -> Union[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
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
        question : torch.Tensor  (vocab_size, n_words_in_sentence)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        idx = str(idx)
        image = Image.open(f"{self.image_dir}/{self.json['image'][idx]}")
        image = self.transform(image)

        if self.answer:
            return image, self.questions[idx], self.answer_tensors[idx], torch.tensor(self.answers[idx])
        else:
            return image, self.questions[idx]

    def __len__(self) -> int:
        return len(self.questions)


class VQAStrQuestionOneHotAnswerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        answer: bool = True,
        onehot_type: Optional[
            Literal[
                "global_mode",
                "most_confident_mode",
                "confidence_weighted_average",
                "global_mode_except_unanswerable",
            ]
        ] = None,
        confidence_weight: Optional[Mapping[str, float]] = None,
    ):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        with open(df_path) as f:
            self.json = json.load(f)  # 画像ファイルのパス，question, answerを持つDataFrame

        self.questions = {}  # (n_questions, len_sentence)
        for k, question in self.json["question"].items():
            self.questions[k] = process_text(question)

        self.answer = answer
        if self.answer:
            self.aidx = all_answers_list(df_path)
            if onehot_type == "global_mode":
                self.answer_tensors = global_mode_tensors(df_path, self.aidx)
            elif onehot_type == "global_mode_except_unanswerable":
                self.answer_tensors = global_mode_except_unanswerable_tensors(df_path, self.aidx)
            elif onehot_type == "most_confident_mode":
                self.answer_tensors = most_confident_mode_tensors(df_path, self.aidx)
            elif onehot_type == "confidence_weighted_average":
                self.answer_tensors = confidence_weighted_average_tensors(
                    df_path,
                    self.aidx,
                    confidence_weight=confidence_weight,
                )
            else:
                raise NoModeTypeError

            self.answers = get_answers_tensor(df_path, self.aidx)

    def __getitem__(self, idx: int) -> Union[
        tuple[
            torch.Tensor,
            str,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[
            torch.Tensor,
            str,
        ],
    ]:
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
        question : torch.Tensor  (vocab_size, n_words_in_sentence)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        idx = str(idx)
        image = Image.open(f"{self.image_dir}/{self.json['image'][idx]}")
        image = self.transform(image)

        if self.answer:
            return image, self.questions[idx], self.answer_tensors[idx], torch.tensor(self.answers[idx]), idx
        else:
            return image, self.questions[idx]

    def __len__(self) -> int:
        return len(self.questions)

    def get_weighted_sampler(
        self,
        count_threshould: float = 1e-1,
        bias: Mapping[str, float] = {},
        exclude: list[str] = [],
    ):
        answers = torch.stack(list(self.answer_tensors.values()))  # (n_questions, n_answer_vocab)

        answer_sum = answers.sum(dim=0)  # (n_answer_vocab,)
        exclude_tensor = torch.full((len(self.questions),), False)  # (n_questions,)
        if exclude:
            for exclude_word in exclude:
                exclude_int = self.aidx.str_to_idx[exclude_word]
                for i in range(answers.shape[1]):
                    exclude_tensor |= self.answers[:, i] == exclude_int

        answer_weights = (answer_sum >= count_threshould) / torch.clamp(answer_sum, min=1e-2)  # (n_answer_vocab,)
        for k, v in bias.items():
            answer_weights[self.aidx.str_to_idx[k]] *= v

        data_weights = torch.mv(answers, answer_weights) * torch.where(exclude_tensor, 0, 1)  # (n_questions,)
        pd.DataFrame(data_weights.numpy()).to_csv("data_weights.csv", index=False, header=None)
        sampler = WeightedRandomSampler(
            data_weights,
            len(self),
            replacement=True,
        )
        return sampler
