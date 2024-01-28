import os
from IPython import display
from IPython.display import Image

HOME = os.getcwd()
print(HOME)

# Pip install method (recommended)
!pip install ultralytics==8.0.196

display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from roboflow import Roboflow

# Create Roboflow instance
rf = Roboflow(api_key="yourapikeyhere")
project = rf.workspace("project-nlr2u").project("sidewalk-segmentation-v4gpn")
dataset = project.version(1).download("yolov8")

"""## Custom Training"""

# Create YOLO instance and perform training
yolo = YOLO(task="segment", mode="train", model="yolov8s-seg.pt", data=f"{dataset.location}/data.yaml", epochs=20, imgsz=640)
yolo.train()

# Display training results
print("Training Results:")
!ls {HOME}/runs/segment/train/

# Display confusion matrix
Image(filename=f'{HOME}/runs/segment/train2/confusion_matrix.png', width=600)

# Display training results image
Image(filename=f'{HOME}/runs/segment/train/results.png', width=600)

# Display validation batch prediction
Image(filename=f'{HOME}/runs/segment/train2/val_batch0_pred.jpg', width=600)

"""## Validate Custom Model"""

# Validate the custom model
yolo_validate = YOLO(task="segment", mode="val", model=f'{HOME}/runs/segment/train/weights/best2.pt', data=f'{dataset.location}/data.yaml')
yolo_validate.run()

"""## Inference with Custom Model"""

# Perform inference with the custom model
yolo_predict = YOLO(task="segment", mode="predict", model=f'{HOME}/runs/segment/train/weights/best2.pt', conf=0.25, source=f'{dataset.location}/test/images', save=True)
yolo_predict.run()

import glob

# Display the first three predicted images
for image_path in glob.glob(f'{HOME}/runs/segment/predict/*.jpg')[:3]:
    display(Image(filename=image_path, height=600))
    print("\n")
