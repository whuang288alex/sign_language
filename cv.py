import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.io import read_image
from torchvision import datasets, transforms
import torchvision.transforms as T
from model import ConvNeuralNet
import matplotlib.pyplot as plt
from PIL import Image
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

def main():

    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    index = 0
    
    while True:
        _, frame = cap.read()
        h, w, c = frame.shape
        capture = False
        
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        
        if k % 256 == 32:
            capture = True

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

                y_width = y_max-y_min
                x_width = x_max-x_min
                y_central = y_min + y_width/2
                x_central = x_min + x_width/2
                
                pad = max(x_width, y_width)/2
                
                cv2.rectangle(frame, (x_central - pad, y_central - pad), (x_central + pad, y_central + pad), (0, 255, 0), 2)
                
                hand_frame = framergb[y_min:y_max, x_min:x_max]
                
                if capture:
                    hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
                    hand_frame = cv2.flip(hand_frame, 1)
                    cv2.imwrite( os.path.join("captured", "hand{}.jpg".format(index)), hand_frame)
                    index += 1
                   
        cv2.imshow("Frame", frame)
        
    cap.release()

main()