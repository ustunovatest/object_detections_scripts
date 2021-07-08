import cv2
import time
cap = cv2.VideoCapture("/home/ustunova/Documents/video.mp4")

while True:
    
    start_time = time.time()
    ret, image = cap.read()
    cv2.imshow("test", image)
    cv2.waitKey(1)
    end_time = time.time()
    #print("Fps: {}".format(1 / (end_time - start_time)))
    print(image.shape)