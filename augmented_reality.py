from yolo import detect_objects;
import cv2
import numpy as np
   
image_path = 'image/sports.jpg' 
img, boxes, confidences, class_ids, class_names, idxs = detect_objects(image_path)

frame_count = 0
step = 0.5
opacity = 1.0

while True:
    image_copy = img.copy()
    
    for i in idxs.flatten():
        x, y, w, h = boxes[i]     
        label = str(class_names[class_ids[i]])
        confidence = confidences[i]

        color = tuple([int((np.sin(frame_count/10 + shift) + 1) * 127) for shift in range(3)])
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 5)

        opacity = abs(np.sin(frame_count * step))
        overlay = image_copy.copy()
        text = f"{label}: {round(confidence * 100, 2)}%"
        text_color = (0, 0, 255)
        text_bg_color = (255, 255, 255)
        text_bg = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(overlay, (x, y-10 - text_bg[1]), 
                    (x + text_bg[0], y-5), text_bg_color, -1)
        cv2.putText(overlay, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, opacity, image_copy, 1 - opacity, 0, image_copy)
        
    frame_count += 1

    cv2.imshow('Augmented Reality Simulation', image_copy)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break 
    