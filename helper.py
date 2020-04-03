"""
Helper functions for the other python scripts.
"""

import numpy as np

def img_transformation(img):
    """
    Transform the input image to the correct format
    to be feed into the Faster RCNN.
    """
    return img