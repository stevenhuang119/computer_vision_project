"""
A python module that implements human detection in an image.
"""

import numpy as np
import torchvision

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
        pass

    def forward(self):
        """
        Compute the objects detected and output a box around the objects.
        """
        pass

    def evaluate(self):
        """
        Evaluate the model with mAP.
        """
        pass