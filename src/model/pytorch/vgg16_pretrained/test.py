import torch
import torchvision.models as models
from ..testing import run_test


def test():	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_classes = 10
	model = models.vgg16(weights=None)
	num_features = model.classifier[-1].in_features
	model.classifier[-1] = torch.nn.Linear(num_features, num_classes)
	model.to(device)
	model.load_state_dict(torch.load('models/pytorch/vgg16_pretrained.pth', weights_only=True))
	run_test(model, "vgg16_pretrained")

if __name__ == '__main__':
    test()