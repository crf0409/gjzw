# Backbone models for benchmarking
from .registry import register_backbone, get_backbone, BACKBONE_REGISTRY, list_backbones

# 原始 9 个 backbone (注册到 BACKBONE_REGISTRY)
from .resnet import ResNet50Classifier
from .vgg import VGG16Classifier, VGG19Classifier
from .inception import InceptionV3Classifier, InceptionResNetV2Classifier
from .efficientnet import EfficientNetB3Classifier
from .mobilenet import MobileNetV3Classifier
from .vit import ViTB16Classifier
from .custom_mlp import CustomMLPClassifier
from .capsnet import CapsNetClassifier

# 论文返修扩量: 5 个 timm SOTA backbone (要 huggingface 联网)
from . import timm_backbones  # noqa: F401  注册副作用

# torchvision 版 modern backbones (避开 hf 联网, cache 内可用)
from . import torchvision_extras  # noqa: F401  注册副作用
