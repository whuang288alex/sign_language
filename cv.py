import numpy as np
import pandas as pd
import cv2
import torch
import mediapipe as mp
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

mphands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mphands.Hands()

from model import 
model = ModelClass()
model.load_state_dict(torch.load("./model.pt"))

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

img_counter = 0
analysisframe = ''
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1) # display a frame for 1 ms
    # press esc
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    # press space
    if k % 256 == 32:

cap.release()
cv2.destroyAllWindows()
