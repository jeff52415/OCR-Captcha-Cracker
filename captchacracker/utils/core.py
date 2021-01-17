import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ImageFeatureExtractor(nn.Module):
    def __init__(self, resnet_model):
        super(ImageFeatureExtractor, self).__init__()
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = ConvLayer(128, 64, 3, (2, 1), 1, 0.3)
        self.layer4 = ConvLayer(64, 64, 3, (2, 1), 1, 0.3)
        self.layer5 = ConvLayer(64, 64, 3, (2, 1), 1, 0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class ImageFeatureExtractorLighter(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractorLighter, self).__init__()
        self.layer1 = ConvLayer(3, 64, 3, (2, 2), 1, 0.3)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.layer2 = ConvLayer(64, 64, 3, (2, 1), 1, 0.3)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.layer3 = ConvLayer(64, 64, 3, (2, 1), 1, 0.3)
        self.layer4 = ConvLayer(64, 64, 3, (2, 1), 1, 0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop=0.3):
        super(BiRNN, self).__init__()
        self.GRU = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=False,
            dropout=drop,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        out, _ = self.GRU(
            x
        )  # out: tensor of shape (seq_length, batch_size, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class CRNN(nn.Module):
    def __init__(self, cnn_backbone, backbone, num_classes=37):
        super(CRNN, self).__init__()
        if backbone == "light":
            self.feature_extractor = ImageFeatureExtractorLighter()
        else:
            self.feature_extractor = ImageFeatureExtractor(cnn_backbone)

        self.GRU_First = BiRNN(64, 256, 2, 256, 0.3)
        self.GRU_Second = BiRNN(256, 256, 2, 256, 0.3)
        self.output = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(2).permute(2, 0, 1)
        x = self.GRU_First(x)
        x = self.GRU_Second(x)
        x = self.output(x)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage)[
                    "model_state_dict"
                ]
            )
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")
