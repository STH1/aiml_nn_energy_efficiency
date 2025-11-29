import torch.nn as nn
from torchvision.models import (
    mobilenet_v2,
    efficientnet_b0,
    alexnet,
    resnet101,
    densenet201
)

def get_mobilenet_v2(pretrained=False, num_classes=1000):
    model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    if num_classes != 1000:
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_efficientnet_b0(pretrained=False, num_classes=1000):
    model = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    if num_classes != 1000:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_alexnet(pretrained=False, num_classes=1000):
    model = alexnet(weights="IMAGENET1K_V1" if pretrained else None)
    if num_classes != 1000:
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_resnet101(pretrained=False, num_classes=1000):
    model = resnet101(weights="IMAGENET1K_V2" if pretrained else None)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_densenet201(pretrained=False, num_classes=1000):
    model = densenet201(weights="IMAGENET1K_V1" if pretrained else None)
    if num_classes != 1000:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
