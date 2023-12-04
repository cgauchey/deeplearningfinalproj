from typing import Iterator
import torch
from pathlib import Path
import cv2, os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# from torchvision.models import resnet34, ResNet34_Weights
# from torchvision.models import resnet18, ResNet18_Weights

# VM crashing with new imports, use old one
from torchvision import models


"""
Inspired by https://github.com/fedeizzo/camera-pose-estimation/blob/master/camera-pose-estimation/model/models/mapnet.py
"""

class ClassyPoseNet(nn.Module):
    
    def __init__(self, feature_dimension: int, dropout_rate: float, num_classes:int, 
                 device: torch.device):

        super().__init__()
        # self.feature_extractor = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # self.feature_extractor = resnet34(pretrained=True)
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_rate = dropout_rate
        self.device = device
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)

        out_feature_extractor = self.feature_extractor.fc.in_features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(out_feature_extractor, feature_dimension),
            nn.ReLU(),
            nn.Linear(feature_dimension, feature_dimension // 2),
            nn.ReLU(),
            nn.Linear(feature_dimension // 2, feature_dimension // 4),
        )

        self.cls_encoder = nn.Linear(feature_dimension // 4, num_classes)
        self.xyz_encoder = nn.Linear(feature_dimension // 4, 3)
        self.rpy_encoder = nn.Linear(feature_dimension // 4, 3)

        init_modules = [
            self.feature_extractor.fc,
            self.cls_encoder,
            self.xyz_encoder,
            self.rpy_encoder,
        ]

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Input shape should be [(Batch*Steps) x Channels x Width x Height]
        """
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        c = self.cls_encoder(x)
        xyz = self.xyz_encoder(x)
        wxyz = self.rpy_encoder(x)

        return torch.cat((c, xyz, wxyz), 1).float32()
    

    def parameters(self):
        # Return only the parameters that require gradients
        # This is the feature extractor fc, and the encoder layers
        return filter(lambda p: p.requires_grad, super().parameters())