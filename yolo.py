import cv2 
import numpy as np

config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'

yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)

def detect_objects(image_path):
    class_names = open('coco.names').read().strip().split('\n')
    
    img = cv2.imread(image_path)
    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)
    ln = yolo.getUnconnectedOutLayersNames()
    layer_outputs = yolo.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maxima Suppression to remove overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    detected_objects = [(class_names[class_ids[i]], boxes[i]) for i in idxs.flatten()]
    for product, box in detected_objects:
        print(f"Detected {product} at {box}")
    
    return img, boxes, confidences, class_ids, class_names, idxs



