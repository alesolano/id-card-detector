import tensorflow as tf
from tensorflow.core.framework import graph_pb2


from infer import evaluate_video




class Run():

    def __init__(self, cpu=True):
        self.cpu = cpu

        self.load_model()


    def load_model(self, model_path='./model/frozen_inference_graph.pb'):

        # Load frozen graph
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        # Input tensor is the image
        self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')

        self.output_tensors = [detection_boxes, detection_scores, detection_classes]

        self.sess = tf.Session(graph=tf.get_default_graph())


    def get_bboxes_from_video(self, video_path, draw=False):

        boxes_per_frame = evaluate_video(self.sess, video_path, self.image_tensor, self.output_tensors, draw=draw)

        return boxes_per_frame

