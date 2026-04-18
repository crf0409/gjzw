# Backbone models for benchmarking
from .registry import register_backbone, get_backbone, BACKBONE_REGISTRY, list_backbones

# Import all backbone models to register them
from .resnet import ResNet50Classifier
from .vgg import VGG16Classifier, VGG19Classifier
from .inception import InceptionV3Classifier, InceptionResNetV2Classifier
from .efficientnet import EfficientNetB3Classifier
from .mobilenet import MobileNetV3Classifier
from .vit import ViTB16Classifier
from .custom_mlp import CustomMLPClassifier
