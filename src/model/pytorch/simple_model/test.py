import torch
from .build import CNN
from ..testing import run_test


def test():	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_classes = 10
	model = CNN(num_classes=num_classes).to(device)
	model.load_state_dict(torch.load('models/pytorch/simple_model.pth', weights_only=True))
	run_test(model, "simple_model")

if __name__ == '__main__':
    test()