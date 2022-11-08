import cv2
import torch
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

cap = cv2.VideoCapture(0)