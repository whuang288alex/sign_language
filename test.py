import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
from model import ConvNeuralNet
import matplotlib.pyplot as plt
from PIL import Image


model = ConvNeuralNet()
model.load_state_dict(torch.load("./model.pt"))
model.eval()

transform =  T.Grayscale()
img =  transform(read_image(os.path.join('captured','img_5.jpg')))

transform =  T.Resize((28,28))
img =  transform(img)

input = torch.unsqueeze(img, 0)
output = model(input.float())
pred = output.max(1, keepdim=True)[1]
print(pred)
