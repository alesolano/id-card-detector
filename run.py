import argparse

import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from infer import evaluate_video


class Run():

    def __init__(self, cpu=True):
        self.cpu = cpu

        self.load_model()


    def load_model(self, model_path='./model/frozen_inference_graph.pb'):

        # Load frozen graph
        tf.compat.v1.reset_default_graph
        graph_def = graph_pb2.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph = tf.compat.v1.get_default_graph()

        # Input tensor is the image
        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')

        self.output_tensors = [detection_boxes, detection_scores, detection_classes]

        self.sess = tf.compat.v1.Session(graph=graph)


    def get_bboxes_from_video(self, video_path, draw=False):

        boxes_per_frame = evaluate_video(self.sess, video_path, self.image_tensor, self.output_tensors, draw=draw)

        return boxes_per_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='path to the video file')
    parser.add_argument('--draw', type=bool, default=False, help='flag to draw the poses and save the images')
    args = parser.parse_args()

    r = Run()
    boxes = r.get_bboxes_from_video(args.video_path, args.draw)
    print(boxes)