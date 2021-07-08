#WORKED ON KEYPOINTS DETECTION(DIRECT TENSORFLOW)
import os
import pathlib
import tensorflow as tf
import time
import cv2
import numpy as np
from PIL import Image
from tensorflow.python.training.tracking.base import no_automatic_dependency_tracking
from tensorflow.python.util.dispatch import dispatch

path_to_saved_model = "/home/ustunova/Downloads/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8/saved_model"
image_np = cv2.imread("/home/ustunova/Downloads/rammstein.jpg")

print("Model loading ... ")
start_time = time.time()
detect_fn = tf.saved_model.load(path_to_saved_model)
end_time = time.time()

print("Done! Took {} seconds ... ".format(end_time - start_time))

def detect(image):

    num_detections = 6

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    
    #print(detections[])
    print("Image Shape: ")
    print(image.shape)

    height, width, _ = image.shape

    for x in detections:
        print(x)
        print(detections[x].shape)

    
    bboxes = detections["detection_boxes"][0][:num_detections]
    for bbox_index in range(len(bboxes)):
        bbox = bboxes[bbox_index]
        xmin = int(bbox[1] * width)
        ymin = int(bbox[0] * height)
        xmax = int(bbox[3] * width)
        ymax = int(bbox[2] * height)

             
        image = cv2.circle(image, (xmin, ymin), 5, (0, 255, 0), 5)
        image = cv2.circle(image, (xmax , ymax), 5, (0, 255, 0), 5)
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
        
        keypoints = detections['detection_keypoints'][0][bbox_index]

        for keypoint in keypoints:
            x, y = int(keypoint[1] * width), int(keypoint[0] * height)
            image = cv2.circle(image, (x, y), 2, (0, 255, 0), 2)


    
    return image


image = detect(image_np)
cv2.imshow("Test", image)
cv2.waitKey(0)


end_time = time.time()
print("Fps: {}".format(end_time - start_time))


