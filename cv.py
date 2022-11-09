import cv2
import time
import os

cap = None
try:    
    cap = cv2.VideoCapture(0) 
except Exception as e:
    print(Exception, e) 

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1) # display a frame for 1 ms
    cv2.imshow('Video', frame)
    
    # press esc
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    
cap.release()
cv2.destroyAllWindows()
