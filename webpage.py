import argparse
import datetime
import os
import threading
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from torchvision.io import read_image
from model import ConvNeuralNet


# initialize variables for the translation system
model = ConvNeuralNet()
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model.load_state_dict(torch.load("./model.pt",  map_location=torch.device('cpu')))
model.eval()


# initialize variables for multithreading
outputFrame = None
lock = threading.Lock()


# initialize a flask object
app = Flask(__name__)
vs = VideoStream(0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


def generate():
    global outputFrame, lock
    while True:
		# wait until the lock is acquired
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


def predict(img):
    # transform the input image
    img = T.Grayscale()(img)
    img = T.Resize((224, 224))(img)
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
    
    # return the predicted character
    return letters[idx]


def translate():
    global outputFrame, lock
    
    hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    total = 0
    while True:

        # process keyboard input
        k = cv2.waitKey(1)
        capture = False
       
        # get the frame and locate hands
        frame = vs.read()
        h, w, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        
        # if there is a hand detected
        if hand_landmarks:
            index = 0
            # for each hand detected (maximum 2)
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                # For each hand, there will be 21 points
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                
                # make sure the hand frame doesn't go out of bound
                x_max = max(w, x_max)
                y_max = max(h, y_max)
                x_min = min(0, x_min)
                y_min = min(0, y_min)
                if y_max - y_min <= 0 or x_max - x_min <= 0:
                    continue
                
                # draw a rectangle around the detected hands and show the captured hands landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, handLMs, mp.solutions.hands.HAND_CONNECTIONS)

                # make prediction based on hand gesture and show the result
                hand_frame = framergb[y_min:y_max, x_min:x_max]
                hand_frame_tensor = torch.permute(torch.from_numpy(hand_frame), (2,0,1))
                frame = cv2.putText(frame,  predict(hand_frame_tensor), (100,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        
        # write the result to 
        with lock:
            outputFrame = frame.copy()
 
# check to see if this is the main thread of execution
if __name__ == '__main__':
    
	# construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="0.0.0.0",
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default = 8000,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

	# start a thread that will perform motion detection
    t = threading.Thread(target=translate)
    t.daemon = True
    t.start()
 
	# start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()