from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer


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
        x = self.relu(self.bn1(self.conv1(x)))  # (*, C=3, H, W) -> (*, 64, H2, W2)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # -> (*, 512, H3, W3)

        x = self.avgpool(x)  # -> (*, 512, 1, 1)
        x = x.view(x.size(0), -1)  # -> (*, 512)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


def ResNet101():  # Out of memory
    return ResNet(BottleneckBlock, [3, 4, 23, 3])


class VQASampleModel(nn.Module):
    """
    配布されたモデル
    画像エンコーダ: ResNet
    質問エンコーダ: One-Hotベクトル + MLP
    """

    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()  #
        self.text_encoder = nn.Sequential(
            nn.Linear(vocab_size, 512),  # (*, vocab_size) -> (*, 512)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer),
        )

    def forward(self, image, question):
        # image: (*, C, H, W)
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


class InvalidResnetType(RuntimeError):
    pass


class VQAEmbeddingModel(nn.Module):
    """
    質問をEmbedしたモデル
    画像エンコーダ: ResNet
    質問エンコーダ: Embedding + LSTM
    """

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


class VQABertEmbeddingModel(nn.Module):
    """
    質問のエンコードにBertを用いたモデル
    画像エンコーダ: ResNet
    質問エンコーダ: BERT
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        net_type: Literal["ResNet18", "ResNet50", "ResNet101", "DenseNet121"],
        n_answer: int,
        device: str,
    ):
        super().__init__()
        if net_type == "ResNet18":
            self.resnet = ResNet18()
        elif net_type == "ResNet50":
            self.resnet = ResNet50()
        elif net_type == "ResNet101":
            self.resnet = ResNet101()
        elif net_type == "DenseNet121":
            self.resnet = models.densenet121(pretrained=False, num_classes=512)
        else:
            raise InvalidResnetType

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
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
