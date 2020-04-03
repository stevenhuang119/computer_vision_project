"""
A python module that implements human detection in an image.
"""

import numpy as np
import torchvision
from PIL import Image

from helper import *

class detector:
    """
    Utilize pre-trained Faster R-CNN Resnet-50 Model

    Input to the pre-trained model:
    1) a tensor of dimension (n, c, h, w)
        i) n is the number of images
        ii) c is the number of channels per image
        iii) h is the height
        iv) w is the width
    2) each image is of at least 800 pixels

    Output of the pre-trained model:
    1) bounding box of all detected objects (N, 4)
    2) labels of the predicted classes
    3) scores of each predicted label
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]


    def make_predictions(self, img_path, threshold=0.8):
        """
        Compute the objects detected and output a box around the objects.
        """
        # obtain the images
        img = Image.open(img_path) 
        img = img_transformation(img)

        # run the image through the RCNN model
        pred = self.model([img]) # only passing one image in for testing
        pred_labels = [self.CATEGORY_NAMES[i] for i in list(
            pred[0]['labels'].numpy()
        )]
        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(
            pred[0]['boxes'].detch().numpy()
        )]
        pred_score = list(pred[0]['scores'].detach().numpy())

        # filter the predictions with scores higher than the threshold
        pred_index_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        pred_boxes = None
        pred_labels = None

        return pred_boxes, pred_labels




    def evaluate(self):
        """
        Evaluate the model with mAP.
        """
        pass