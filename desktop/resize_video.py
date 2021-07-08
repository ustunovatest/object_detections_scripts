import cv2
import time


video_path = "/home/ustunova/Documents/mfd_centernet_keypoint_detection_v1/error_videos_2/DJI_0096.MP4"
cap = cv2.VideoCapture(video_path)

ret = True
images = []
try:
    while ret:
        
        start_time = time.time()
        ret, image = cap.read()
        print(ret)
        image = cv2.resize(image, (640, 480))
        images.append(image)
        cv2.imshow("test", image)
        end_time = time.time()
        #print("Fps: {}".format(1 / (end_time - start_time)))
        #print(image.shape)
except:
    None

out = cv2.VideoWriter("/home/ustunova/Documents/test.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (640, 480))
for i in range(len(images)):
    out.write(images[i])

print("Done ... ")
out.release()