import cv2
from yolo import detect_objects

image_path = 'image/sports.jpg'
def box_n_text(image_path) :
    img, boxes, confidences, class_ids, class_names, idxs = detect_objects(image_path)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i] 
            color = (0, 255, 0)
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

img = box_n_text(image_path)
cv2.imshow("Object Detection", img )
cv2.waitKey(0)
cv2.destroyAllWindows()