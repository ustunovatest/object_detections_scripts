import os
import numpy as np
from six import b
import tensorflow as tf
import cv2
import time
import statistics

from object_detection.utils import label_map_util
from object_detection.utils import config_util


def detect(interpreter, input_tensor, include_keypoint=False):

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  num_detections = interpreter.get_tensor(output_details[3]['index'])
  if include_keypoint:
    kpts = interpreter.get_tensor(output_details[4]['index'])
    kpts_scores = interpreter.get_tensor(output_details[5]['index'])
    return boxes, classes, scores, num_detections, kpts, kpts_scores
  else:
    return boxes, classes, scores, num_detections, None, None



def draw_boxes(image, boxes=None, box_scores=None, keypoints=None, keypoint_scores=None):
    height, width, _ = image.shape
    
    for index, box in enumerate(boxes[0]):
        if box_scores[0][index] >= 0.5:
            xmin = int(box[1] * width)
            ymin = int(box[0] * height)
            xmax = int(box[3] * width)
            ymax = int(box[2] * height)

            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            for kp_index, kp in enumerate(keypoints[0][index]):
                if keypoint_scores[0][index][kp_index] >= 0.3:
                    x = int(kp[1] * width)
                    y = int(kp[0] * height)

                    image = cv2.circle(image, (x, y), 2, (0, 255, 0), 2)

    return image
    


if __name__ == "__main__":
    model_path = '/home/ustunova/Documents/mfd_centernet_keypoint_detection_v1/pretrained_models/centernet_mobilenetv2fpn_512x512_coco17_kpts/centernet_mobilenetv2_fpn_kpts/model.tflite'
    label_map_path = '/home/ustunova/Documents/mfd_centernet_keypoint_detection_v1/pretrained_models/centernet_mobilenetv2fpn_512x512_coco17_kpts/centernet_mobilenetv2_fpn_kpts/label_map.txt'
    
    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    end_time = time.time()
    print("Model loaded {} seconds".format(end_time - start_time))
    cap = cv2.VideoCapture(0)
    
    fpsses = []

    while True:
      
      start_time = time.time()
      ret, image_numpy = cap.read()
      
      input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.float32)
      input_tensor = tf.image.resize(input_tensor, (320, 320))
      input_tensor = input_tensor[tf.newaxis, ...]

      boxes, classes, scores, num_detections, kpts, kpts_scores = detect(interpreter, input_tensor, include_keypoint=True)
      image_numpy = draw_boxes(image_numpy, boxes=boxes, box_scores=scores, keypoints=kpts, keypoint_scores=kpts_scores)
      
      cv2.imshow("test", image_numpy)
      pressed_key = cv2.waitKey(1)
      if pressed_key & 0xFF == ord('q'):
        break
      end_time = time.time()
      fpsses.append(1 / (end_time - start_time))
    
    
    print("Mean Fps: {}".format(statistics.mean(fpsses)))