#!/usr/bin/python3

import sys
import time
import math
import cv2
from ultralytics import YOLO
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk


if __name__ == '__main__':

    # Load a pretrained YOLOv8n model
    model = YOLO('best2.pt')

    # Set webcam size
    width, height = 540, 360

    # use video file
    cap = cv2.VideoCapture('vid.mp4')

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)
        
        # print(motiontime)
        # print(state.imu.rpy[0])
        # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        # print(state.imu.rpy[0])
        # print(state.imu.accelerometer)
        # print(state.imu.temperature)

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        # cmd.mode = 2
        # cmd.gaitType = 1
        # # cmd.position = [1, 0]
        # # cmd.position[0] = 2
        # cmd.velocity = [-0.2, 0] # -1  ~ +1
        # cmd.yawSpeed = 0
        # cmd.bodyHeight = 0.1

        # if(motiontime > 0 and motiontime < 1000):
        #     cmd.mode = 1
        #     cmd.euler = [-0.3, 0, 0]
        
        # if(motiontime > 1000 and motiontime < 2000):
        #     cmd.mode = 1
        #     cmd.euler = [0.3, 0, 0]
        
        # if(motiontime > 2000 and motiontime < 3000):
        #     cmd.mode = 1
        #     cmd.euler = [0, -0.2, 0]
        
        # if(motiontime > 3000 and motiontime < 4000):
        #     cmd.mode = 1
        #     cmd.euler = [0, 0.2, 0]
        
        # if(motiontime > 4000 and motiontime < 5000):
        #     cmd.mode = 1
        #     cmd.euler = [0, 0, -0.2]
        
        # if(motiontime > 5000 and motiontime < 6000):
        #     cmd.mode = 1
        #     cmd.euler = [0.2, 0, 0]
        
        # if(motiontime > 6000 and motiontime < 7000):
        #     cmd.mode = 1
        #     cmd.bodyHeight = -0.2
        
        # if(motiontime > 7000 and motiontime < 8000):
        #     cmd.mode = 1
        #     cmd.bodyHeight = 0.1
        
        # if(motiontime > 8000 and motiontime < 9000):
        #     cmd.mode = 1
        #     cmd.bodyHeight = 0.0
        
        # if(motiontime > 9000 and motiontime < 11000):
        #     cmd.mode = 5
        
        # if(motiontime > 11000 and motiontime < 13000):
        #     cmd.mode = 6
        
        # if(motiontime > 13000 and motiontime < 14000):
        #     cmd.mode = 0
        
        # if(motiontime > 1500 and motiontime < 2000):
        #     cmd.mode = 2
        #     cmd.gaitType = 1
        #     cmd.velocity = [-0.2, 0] # -1  ~ +1
        #     # cmd.yawSpeed = 2
        #     cmd.footRaiseHeight = 0.1
        #     print("walk")

        # if(motiontime > 2000 and motiontime < 2500):
        #     cmd.mode = 2
        #     cmd.gaitType = 1
        #     cmd.velocity = [0, 0] # -1  ~ +1
        #     cmd.yawSpeed = 0
        #     cmd.footRaiseHeight = 0.1
        #     print("walk-side")
        
        # if(motiontime > 2500 and motiontime < 20000):
        #     cmd.mode = 0
        #     cmd.velocity = [0, 0]
        
        # if(motiontime > 20000 and motiontime < 24000):
        #     cmd.mode = 2
        #     cmd.gaitType = 1
        #     cmd.velocity = [0.2, 0] # -1  ~ +1
        #     cmd.bodyHeight = 0.1
        #     # printf("walk\n")
                        
        ret, frame = cap.read()

        # Resize the frame
        frame = cv2.resize(frame, (width,height))

        # Perform YOLO inference on the frame
        results = model.predict(frame, conf=0.85)

        
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs

        
        # Convert tensor to NumPy array
        boxes_np = boxes.xyxy.cpu().numpy()

        if boxes_np.size == 0: continue
        
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
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0, 0.2] # -1  ~ +1
            cmd.yawSpeed = 0
            cmd.footRaiseHeight = 0.1
            print("walk-left")

        elif dx < -10:
            cv2.putText(frame, 'right', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0, -0.2] # -1  ~ +1
            cmd.yawSpeed = 0
            cmd.footRaiseHeight = 0.1
            print("walk-right")

        else:
            cv2.putText(frame, 'centre', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        # draw a vertical line at the center of the image
        cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('Webcam YOLO', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

        udp.SetSend(cmd)
        udp.Send()

    cap.release()
    cv2.destroyAllWindows()


# Release the webcam and close all OpenCV windows


