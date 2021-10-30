import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


MODEL_NAME = 'model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load the Tensorflow model into memory.
tf.reset_default_graph()
from tensorflow.core.framework import graph_pb2
graph_def = graph_pb2.GraphDef()
with open("./model/frozen_inference_graph.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')


# video path
video_path = 'veriff1.mp4'
frame_provider = VideoReader(video_path)

img_i = -1
skip = 100


with tf.Session() as sess:
    for image in frame_provider:

        img_i += 1
        #if img_i % skip != 0: continue
        if (img_i != 130) and (img_i != 200): continue

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        print(1, image.shape)

        # Draw the results of the detection (aka 'visulaize the results')
        image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.35)

        boxes = boxes[scores > 0.35]
        print(scores[scores > 0.1])
        for box in boxes:
            y_min, x_min, y_max, x_max = box
            print(img_i)
            print(x_min, y_min, x_max, y_max)
        
        #print(image.shape)
        
        img_path = f'./{img_i:05}.png'
        cv2.imwrite(img_path, image)