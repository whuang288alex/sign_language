import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.io import read_image
from torchvision import datasets, transforms
import torchvision.transforms as T
from model import ConvNeuralNet
from dataset import CustomDataset
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp


model = ConvNeuralNet()
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model.load_state_dict(torch.load("./model.pt"))
model.eval()

test_df = CustomDataset("./sign_lang_mnist/sign_mnist_test.csv",transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_df, batch_size=32, shuffle=False)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = correct / len(test_loader.dataset)
    # print("test accuracy:", test_acc)
    return test_acc

def predict(img):
    
    # transform the input image
    transform =  T.Grayscale()
    img =  transform(img)
    transform =  T.Resize((28,28))
    img =  transform(img)
    input = torch.unsqueeze(img, 0)
    
    # predict based on the net
    output = model(input.float())
    pred = output.max(1, keepdim=True)[1]
    idx = pred.data[0][0].item()   
    return letters[idx]

def main():
    
    # test_model(model, test_loader)
    img = read_image(os.path.join("captured","hand0.jpg"))
    letter = predict(img)
    
    return
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        h, w, c = frame.shape
        capture = False
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        
        if k % 256 == 32:
            capture = True
            

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        
        if hand_landmarks:
            index = 0
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
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                hand_frame = framergb[y_min:y_max, x_min:x_max]
                if capture:
                    cv2.imwrite( os.path.join("captured", "hand{}.jpg".format(index)), hand_frame)
                    index += 1
                   
        cv2.imshow("Frame", frame)
        
    cap.release()

if __name__ ==  '__main__':
    main()

