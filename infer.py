import cv2
import numpy as np

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


def evaluate_video(sess, video_path, image_tensor, output_tensors, skip=10, thresh=0.3, draw=False):

    boxes_per_frame = {}
    
    frame_provider = VideoReader(video_path)

    detection_boxes, detection_scores, detection_classes = output_tensors

    img_i = -1
    for image in frame_provider:

        img_i += 1
        if img_i % skip != 0: continue

        image_expanded = np.expand_dims(image, axis=0)

        if draw:
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes) = sess.run(
                [detection_boxes, detection_scores, detection_classes],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index=None,
                agnostic_mode=True,
                use_normalized_coordinates=True,
                line_thickness=3,
                min_score_thresh=thresh)

            img_path = f'./{img_i:05}.png'
            cv2.imwrite(img_path, image)
            print(f'Saved image in {img_path}')

        else:
            (boxes, scores) = sess.run(
                [detection_boxes, detection_scores],
                feed_dict={image_tensor: image_expanded})

        boxes = boxes[scores > thresh]
        print(img_i, scores[scores > 0.1])
        
        # Reorder
        boxes_per_frame[img_i] = []
        for box in boxes:
            y_min, x_min, y_max, x_max = box.tolist()
            boxes_per_frame[img_i].append([x_min, y_min, x_max, y_max])
        
    return boxes_per_frame