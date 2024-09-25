import torch
from torchvision import transforms
from PIL import Image


def get_pred_class(pred_value):
	mapping = {0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep', 8: 'spider', 9: 'squirrel'}
	return mapping[pred_value]

def image_test(model, image_path, f):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	img_height, img_width = 224, 224

	image = Image.open(image_path)

	transform = transforms.Compose([
	    transforms.Resize((img_height, img_width)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	input_tensor = transform(image)
	input_batch = input_tensor.unsqueeze(0)
	input_batch = input_batch.to(device)

	model.eval()

	with torch.no_grad():
	    output = model(input_batch)

	_, predicted_class = torch.max(output, 1)

	actual_class = image_path.split('/')[1].split('_')[0]
	predicted_class = get_pred_class(predicted_class.item())

	print(f"Actual class: {actual_class}, Predicted class: {predicted_class}", file=f)

def run_test(model, key):
	with open(f'reports/pytorch/{key}/test_result.txt', 'w') as f:
		print(f"Model: {key}", file=f)
		test_images = [
			"test/butterfly_test.jpeg",
			"test/butterfly_test_2.jpg",
			"test/cat_test.jpeg",
			"test/cat_test_2.jpg",
			"test/chicken_test.jpeg",
			"test/chicken_test_2.jpg",
			"test/cow_test.jpg",
			"test/cow_test_2.jpg",
			"test/dog_test.jpeg",
			"test/dog_test_2.jpg",
			"test/elephant_test.jpeg",
			"test/elephant_test_2.jpeg",
			"test/horse_test.jpg",
			"test/horse_test_2.jpg",
			"test/sheep_test.jpg",
			"test/sheep_test_2.jpg",
			"test/spider_test.jpg",
			"test/spider_test_2.jpg",
			"test/squirrel_test.jpeg",
			"test/squirrel_test_2.jpeg",
		]
		for test_image in test_images:
			image_test(model, test_image, f)