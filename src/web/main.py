import os.path
import streamlit as st
import io
import torch
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


def customize_theme():
    custom_css = """
        <style>
            html, body, [class*="css"]  {
                font-size: 20px !important;
            }

            h1 {
                font-size: 34px !important;
            }
            h2 {
                font-size: 26px !important;
            }
            h3 {
                font-size: 24px !important;
            }
            p {
                font-size: 22px !important;
            }
            div {
                font-size: 20px !important;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def prepare_image(image_bytes):
	image_stream = io.BytesIO(image_bytes)

	image = Image.open(image_stream).convert("RGB")

	img_height, img_width = 224, 224

	transform = transforms.Compose([
	    transforms.Resize((img_height, img_width)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	input_tensor = transform(image)
	input_tensor = input_tensor.unsqueeze(0)

	return input_tensor

def predict_class(model, image_tensor):
	model.eval()

	with torch.no_grad():
	    output = model(image_tensor)

	_, predicted_class = torch.max(output, 1)

	mapping = {0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep', 8: 'spider', 9: 'squirrel'}

	return mapping[predicted_class.item()]

@st.cache_resource(ttl='1d')
def load_model():
	num_classes = 10
	model = models.vgg16(weights=None)
	num_features = model.classifier[-1].in_features
	model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

	if os.path.isfile('models/pytorch/vgg16_pretrained.pth'):
		model_path = 'models/pytorch/vgg16_pretrained.pth'
	else:
		model_path = hf_hub_download(repo_id="VladKKKKK/animal_classification", filename="vgg16_pretrained.pth")
		
	model.load_state_dict(torch.load(
		model_path,
		weights_only=True,
		map_location = torch.device('cpu')
	))

	return model

def run():

	customize_theme()

	model = load_model()

	st.write(
		"<h1 style='text-align: center;'>Animal class predictor</h1>",
		unsafe_allow_html=True
	)
	
	allowed_formats = ['png', 'jpg', 'jpeg']
	image = st.file_uploader("Upload image", type=allowed_formats)

	_, center_col, _ = st.columns([1, 3, 1])
	if image and center_col.button('Predict animal class', use_container_width=True):

		image_tensor = prepare_image(image.getvalue())
		predicted_class = predict_class(model, image_tensor)

		st.write(
            f"<h2 style='text-align: center;'>Predicted class: {predicted_class.title()}</h1>",
            unsafe_allow_html=True
        )