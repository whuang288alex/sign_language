import os
import torch
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
from model import ConvNeuralNet
import matplotlib.pyplot as plt
import mediapipe as mp

model = ConvNeuralNet()
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model.load_state_dict(torch.load("./model.pt",  map_location=torch.device('cpu')))
model.eval()

def predict(img):
    
    # transform the input image
    transform = T.Resize((224, 224))
    img = transform(img)
    img = img.numpy()
    img = torch.FloatTensor(img)
    img = torch.unsqueeze(img, 0)
    img /= 255.
    
    # predict based on the net
    output = model(img)
    predicted = torch.softmax(output,dim=1) 
    _, predicted = torch.max(predicted, 1) 
    predicted = predicted.data.cpu() 

    idx = predicted[0]
    return letters[idx]

    
img = read_image(os.path.join("captured","handc.jpg"))
letter = predict(img)
print(letter)