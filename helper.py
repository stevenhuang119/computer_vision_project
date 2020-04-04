"""
Helper functions for the other python scripts.
"""

import numpy as np
import torchvision.transforms as T

def img_transformation(img):
    """
    Transform the input image to the correct format
    to be feed into the Faster RCNN.
    """
    transform = T.Compose([T.ToTensor()])
    return transform(img)