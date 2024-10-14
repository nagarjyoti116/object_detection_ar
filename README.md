# Object Detection and Augmented Reality

This project demonstrates how to use OpenCV-Python for object detection and augmented reality. The object detection model is based on YOLO (You Only Look Once), which provides fast and accurate results, while OpenCV is used to display Augmented Reality elements on top of detected objects.

## Prerequisites

Before running this project, ensure you have the following:

* Python 3.7 or later
* OpenCV 4.5 or later
* YOLOv3 weights and configuration files(.weights and .cfg files)
* COCO dataset class names (coco.names)

## Libraries:

* OpenCV: opencv-python
* NumPy: numpy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nagarjyoti116/object_detection_ar.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download Yolo3.weight using the link below
   https://pjreddie.com/media/files/yolov3.weights
5. You can use your own image folder by simply updating the image path accordingly.
 

## Usage

To run object detection and AR integration, execute the following script:

```bash
python yolo.py
python object_detection.py
python bounding_box.py
python augmented_reality.py
```

## Object Detection

The object detection process follows these steps:

* Load the YOLO model using the provided configuration and weights.
* Load the image using opencv and convert it into blob.
* Pass the blob through the YOLO model to detect objects.
* Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
* Output the detected objects class name with their bounding box coordinates

## Augmented Reality Integration

After detecting objects, AR elements are overlaid on top of the identified bounding boxes.
In this project, the following AR features are implemented:

* Bounding boxes are drawn around the detected objects and label them with class names using opencv.
* The color of the bounding box dynamically changes with time using a sine wave function to create a smooth color transition.
* The label and confidence are displayed on a background rectangle.
* The semi-transparent overlay is applied over the original bounding box creating fade effect. Changing the fade step can convert fade effect into pulsating effect.
