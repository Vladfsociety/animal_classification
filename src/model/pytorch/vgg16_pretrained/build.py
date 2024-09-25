import torch
import torchvision.models as models


def VGG16_pretrained(num_classes):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

    return model