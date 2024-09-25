from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_train_test_datasets():
	batch_size = 64
	img_height, img_width = 224, 224

	transform = transforms.Compose([
	    transforms.Resize((img_height, img_width)),
	    transforms.RandomRotation(degrees=20),
	    transforms.RandomHorizontalFlip(p=0.5),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	dataset = datasets.ImageFolder('data', transform=transform)

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

	return train_loader, test_loader