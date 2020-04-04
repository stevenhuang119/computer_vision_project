"""
A python module that implements human detection in an image.
"""
from PIL import Image
import torchvision
from helper import img_transformation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cv2 import cv2

class Detector:
    """
    Utilize pre-trained Faster R-CNN Resnet-50 Model for object detection;

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
        print('model evaluating')
        self.model.eval()
        print('end of model evluation')
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


    def make_predictions(self, img_path, threshold=0.5):
        """
        Compute the objects detected and output a box around the objects.
        """
        # obtain the images
        img = Image.open(img_path) 
        img = img_transformation(img)

        # run the image through the RCNN model
        print('----model predicting----')
        pred = self.model([img]) # only passing one image in for testing
        print('----prediction finishes---')
        pred_labels = [self.CATEGORY_NAMES[i] for i in list(
            pred[0]['labels'].numpy()
        )]
        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(
            pred[0]['boxes'].detach().numpy()
        )]
        pred_score = list(pred[0]['scores'].detach().numpy())

        # filter the predictions with scores higher than the threshold
        pred_index_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        pred_boxes = pred_boxes[:pred_index_t+1]
        pred_labels = pred_labels[:pred_index_t+1]

        return pred_boxes, pred_labels


    def object_detection_pipeline(self, img_path, threshold=0.5,
                                  rect_th=3, text_size=3, text_th=3):
        """
        Establish an entire pipeline where the image is taken in, processed
        by the model, results being interpreted, and boxes drawn
        """
        boxes, labels = self.make_predictions(img_path, threshold)
        img = cv2.imread(img_path)
        cv2.imwrite('./justuploaded.png', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        cv2.imwrite('./converted.png', img)
        for i in range(len(boxes)):
            cv2.rectangle(img,
                          (int(boxes[i][0]), int(boxes[i][1])),
                          (int(boxes[i][2]), int(boxes[i][3])),
                          (0,255,0),
                          thickness=rect_th) # draw the boxese
            cv2.putText(img, labels[i],
                        (int(boxes[i][0]), int(boxes[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, (0,255,0), thickness=text_th)
        # plt.figure(figsize=(20,30))
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        cv2.imwrite('./result.png', img)



def main():
    detector = Detector()
    detector.object_detection_pipeline('./girl_cars.jpg', rect_th=15, text_th=7,
                                       text_size=5, threshold=0.9)



if __name__ == "__main__":
    main()