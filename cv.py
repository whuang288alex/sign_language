import cv2
import time
import os

cap = None
try:    
    cap = cv2.VideoCapture(0) 
except Exception as e:
    print(Exception, e) 

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print(os.path.join(os.getcwd(),"captured"))

counter = 0
while True:
    
    _, frame = cap.read()
    k = cv2.waitKey(1) # display a frame for 1 ms
    cv2.imshow('Video', frame)
    
    # press esc
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    
    if k % 256 == 32:
        path = os.path.join(os.getcwd(),"captured")
        file = os.path.join(path, "img_{}.jpg".format(counter))
        cv2.imwrite(file, frame)
        counter += 1
        print("Captured")
        
    
cap.release()
cv2.destroyAllWindows()
