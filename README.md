# Animal Classification with PyTorch and TensorFlow
This repository contains implementations of several Convolutional Neural Network (CNN) models for animal classification using PyTorch and TensorFlow. The goal is to compare the performance of different architectures and frameworks on this task.
A simple Streamlit demo has been created where the user can upload an image and test the performance of the best model (VGG16 pretrained PyTorch model).
## Dataset
The dataset consists of 10 animal classes, with a total of about 25,000 images. You can view the dataset via this [link](https://www.kaggle.com/datasets/alessiocorrado99/animals10).
Available animal classes:
+ Butterfly
+ Cat
+ Chicken
+ Cow
+ Dog
+ Elephant
+ Horse
+ Sheep
+ Spider
+ Squirrel
## Models
Three different architectures were used, each of which was created in both PyTorch and TensorFlow, resulting in six models. 
+ Simple CNN architecture
```
Input (3 channels) -> [224x224x3]
v
Conv2d(3 -> 16, 3x3) -> ReLU -> MaxPool(2x2) -> [112x112x16]
v
Conv2d(16 -> 32, 3x3) -> ReLU -> MaxPool(2x2) -> [56x56x32]
v
Conv2d(32 -> 64, 3x3) -> ReLU -> MaxPool(2x2) -> [28x28x64]
v
Conv2d(64 -> 128, 3x3) -> ReLU -> MaxPool(2x2) -> [14x14x128]
v
Conv2d(128 -> 256, 3x3) -> ReLU -> MaxPool(2x2) -> [7x7x256]
v
Flatten -> [12544]
v
Linear(12544 -> 512) -> [512]
v
Dropout(0.2)
v
Linear(512 -> 10) -> [10]
```
+ VGG11 architecture implemented from scratch
```
Input (3 channels) -> [224x224x3]
v
Conv2d(3 -> 64, 3x3) -> ReLU -> MaxPool(2x2) -> [112x112x64]
v
Conv2d(64 -> 128, 3x3) -> ReLU -> MaxPool(2x2) -> [56x56x128]
v
Conv2d(128 -> 256, 3x3) -> ReLU -> Conv2d(256 -> 256, 3x3) -> ReLU -> MaxPool(2x2) -> [28x28x256]
v
Conv2d(256 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> MaxPool(2x2) -> [14x14x512]
v
Conv2d(512 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> MaxPool(2x2) -> [7x7x512]
v
Flatten -> [25088]
v
Linear(25088 -> 4096) -> [4096]
v
Dropout(0.5)
v
Linear(4096 -> 4096) -> [4096]
v
Dropout(0.5)
v
Linear(4096 -> 10) -> [10]
```
+ VGG16 with pretrained on imagenet dataset weights
```
Input (3 channels) -> [224x224x3]
v
Conv2d(3 -> 64, 3x3) -> ReLU -> Conv2d(64 -> 64, 3x3) -> ReLU -> MaxPool(2x2) -> [112x112x64]
v
Conv2d(64 -> 128, 3x3) -> ReLU -> Conv2d(128 -> 128, 3x3) -> ReLU -> MaxPool(2x2) -> [56x56x128]
v
Conv2d(128 -> 256, 3x3) -> ReLU -> Conv2d(256 -> 256, 3x3) -> ReLU -> Conv2d(256 -> 256, 3x3) -> ReLU -> MaxPool(2x2) -> [28x28x256]
v
Conv2d(256 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> MaxPool(2x2) -> [14x14x512]
v
Conv2d(512 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> Conv2d(512 -> 512, 3x3) -> ReLU -> MaxPool(2x2) -> [7x7x512]
v
Flatten -> [25088]
v
Linear(25088 -> 4096) -> [4096]
v
Dropout(0.5)
v
Linear(4096 -> 4096) -> [4096]
v
Dropout(0.5)
v
Linear(4096 -> 10) -> [10]
```
## Results
Each model was trained on the animal classification dataset, with the following results tracked:
+ Accuracy: Monitored for both training and validation datasets
+ Loss: Recorded for training and validation datasets

The results include:
+ Accuracy and loss graphs for all models (*accuracy.jpg*, *loss.jpg* files in **reports/pytorch/{model}**, **reports/tensorflow/{model}** folders)
+ Tests on 20 test images with actual vs predicted class (*test_result.txt* files in **reports/pytorch/{model}**, **reports/tensorflow/{model}** folders)

The table below presents the rounded accuracy and loss results for the training and validation datasets for all models:

| Model                         | Final train accuracy | Final train loss | Final validation accuracy | Final validation loss |
|-------------------------------|----------------------|------------------|---------------------------|-----------------------|
| Simple CNN (PyTorch)          | 80%                  | 0.6              | 77%                       | 0.7                   |
| Simple CNN (TensorFlow)       | 88%                  | 0.37             | 79%                       | 0.83                  |
| VGG11 (PyTorch)               | 92.5%                | 0.2              | 81%                       | 0.77                  |
| VGG11 (TensorFlow)            | 93%                  | 0.2              | 82%                       | 0.75                  |
| VGG16 pretrained (PyTorch)    | 98.2%                | 0.06             | 95%                       | 0.17                  |
| VGG16 pretrained (TensorFlow) | 96.5%                | 0.12             | 93.5%                     | 0.24                  |

