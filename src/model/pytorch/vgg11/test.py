import torch
from .build import VGG11
from ..testing import run_test


def test():	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_classes = 10
	model = VGG11(num_classes=num_classes).to(device)
	model.load_state_dict(torch.load('models/pytorch/vgg11.pth', weights_only=True))
	run_test(model, "vgg11")

if __name__ == '__main__':
    test()