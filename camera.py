

import cv2
from ultralytics import YOLO
import numpy as np

import cv2
import numpy as np
import urllib.request as ur

# Load a pretrained YOLOv8n model
model = YOLO('best2.pt')

# Set webcam size
width, height = 540, 360

# cap = cv2.VideoCapture('vidtest3.mp4')
url = "http://192.168.9.84:4040/shot.jpg"

while True:
    # Capture frame-by-frame
    # ret, frame = cap.read()
    
    imgResp = ur.urlopen(url)
    imgnp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    frame = cv2.imdecode(imgnp, -1)

    # Resize the frame
    frame = cv2.resize(frame, (width,height))

    # Perform YOLO inference on the frame
    results = model.predict(frame,show=True, conf=0.75)
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs

    # Convert tensor to NumPy array
    boxes_np = boxes.xyxy.cpu().numpy()
    
    if len(boxes_np) == 0:
        continue
    
    cv2.rectangle(frame, (int(boxes_np[0, 0]), int(boxes_np[0, 1])), (int(boxes_np[0, 2]), int(boxes_np[0, 3])), (0, 255, 0), 2)

    # draw a vertical line from the center of the box
    cv2.line(frame, (int(boxes_np[0, 0] + boxes_np[0, 2] / 2), int(boxes_np[0, 1])),
            (int(boxes_np[0, 0] + boxes_np[0, 2] / 2), int(boxes_np[0, 3])), (255, 255, 255), 2)

    # draw a horizontal line from the center of the box
    cv2.line(frame, (int(boxes_np[0, 0]), int(boxes_np[0, 1] + boxes_np[0, 3] / 2)),
            (int(boxes_np[0, 2]), int(boxes_np[0, 1] + boxes_np[0, 3] / 2)), (255, 255, 255), 2)
    
    # get the centre of the box    
    x,y = int(boxes_np[0, 0] + boxes_np[0, 2] / 2), int(boxes_np[0, 1] + boxes_np[0, 3] / 2)
    
    # get the centre of the image
    cx, cy = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
    
    # get the difference between the box centre and the image centre
    dx, dy = cx - x, cy - y
    
    # write left or right depending on the difference between the box centre and the image centre
    if dx > 10:
        cv2.putText(frame, 'left', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif dx < -10:
        cv2.putText(frame, 'right', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'center', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    # draw a vertical line at the center of the image
    cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Webcam YOLO', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



