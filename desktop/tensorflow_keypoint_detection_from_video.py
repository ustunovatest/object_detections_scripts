import os
import pathlib
import tensorflow as tf
import time
import cv2
import numpy as np
from PIL import Image
from tensorflow.python.training.tracking.base import no_automatic_dependency_tracking
from tensorflow.python.util.dispatch import dispatch
import statistics

#path_to_saved_model = "/home/ustunova/Downloads/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8/saved_model"
path_to_saved_model = "/home/ustunova/Documents/mfd_centernet_keypoint_detection_v1/saved_models/6_points_mobilenet_120k/saved_model"
cap = cv2.VideoCapture("/home/ustunova/Documents/mfd_centernet_keypoint_detection_v1/error_videos_2/DJI_0095.MP4")


print("Model loading ... ")
start_time = time.time()
detect_fn = tf.saved_model.load(path_to_saved_model)
end_time = time.time()

print("Done! Took {} seconds ... ".format(end_time - start_time))

def detect(image, num_detections=1, bbox_threshold=0.5, keypoint_threshold=0.3):
    ımage = cv2.resize(image, (512, 512))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    
    #print(detections[])
    #print("Image Shape: ")
    #print(image.shape)

    height, width, _ = image.shape

    bboxes = detections["detection_boxes"][0][:num_detections]
    bbox_scores = detections['detection_scores'][0][:num_detections]
    for bbox_index in range(len(bboxes)):
        if bbox_scores[bbox_index] >= bbox_threshold:
            bbox = bboxes[bbox_index]
            xmin = int(bbox[1] * width)
            ymin = int(bbox[0] * height)
            xmax = int(bbox[3] * width)
            ymax = int(bbox[2] * height)

                
            image = cv2.circle(image, (xmin, ymin), 5, (0, 255, 0), 5)
            image = cv2.circle(image, (xmax , ymax), 5, (0, 255, 0), 5)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            
            
            keypoints = detections['detection_keypoints'][0][bbox_index]
            keypoint_scores = detections['detection_keypoint_scores'][0][bbox_index]

            for keypoint_index in range(len(keypoints)):
                #print(keypoint_score)
                if keypoint_scores[keypoint_index] >= keypoint_threshold:
                    keypoint = keypoints[keypoint_index]
                    x, y = int(keypoint[1] * width), int(keypoint[0] * height)
                    image = cv2.circle(image, (x, y), 2, (0, 0, 255), 2)
            
    return image

fpsses = []
while True:
    start_time = time.time()
    ret, image = cap.read()
    image = detect(image, num_detections = 4)
    cv2.imshow("test", image)
    pressed_key = cv2.waitKey(1)
    if pressed_key & 0xFF == ord('q'):
        break

    end_time = time.time()
    fpsses.append(1 / (end_time - start_time))
print("Average FPS: {}".format(statistics.mean(fpsses)))