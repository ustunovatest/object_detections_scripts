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

#path_to_saved_model = "/home/ustunova/Downloads/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8/saved_model"
path_to_saved_model = "/home/ustunova/Documents/object_detections_scripts/pretrained_models/centernet_mobilenetv2_fpn_kpts/saved_model/saved_model"
cap = cv2.VideoCapture("/home/ustunova/Documents/test.avi")


print("Model loading ... ")
start_time = time.time()
detect_fn = tf.saved_model.load(path_to_saved_model)
end_time = time.time()

print("Done! Took {} seconds ... ".format(end_time - start_time))

def detect(image, num_detections=1, keypoint_threshold=0.3):
    ımage = cv2.resize(image, (512, 512))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    
    #print(detections[])
    #print("Image Shape: ")
    #print(image.shape)

    height, width, _ = image.shape

    """
    for x in detections:
        print(x)
        print(detections[x].shape)
    """
    
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

        for keypoint_index in range(len(keypoints)):
            keypoint_score = detections['detection_keypoint_scores'][0][bbox_index][keypoint_index]
            #print(keypoint_score)
            if keypoint_score >= keypoint_threshold:
                keypoint = keypoints[keypoint_index]
                x, y = int(keypoint[1] * width), int(keypoint[0] * height)
                image = cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
        


    
    return image


while True:
    start_time = time.time()
    ret, image = cap.read()
    image = detect(image, num_detections = 1)
    cv2.imshow("test", image)
    pressed_key = cv2.waitKey(1)
    if pressed_key & 0xFF == ord('q'):
        break

    end_time = time.time()
    print("FPS: {}".format(1 / (end_time - start_time)))

