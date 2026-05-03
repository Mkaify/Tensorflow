import torch
from PIL import Image
import requests
 
# Load YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load pretrained YOLOv5 small model
 
# Sample image for object detection
image_url = "https://ultralytics.com/images/zidane.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
 
# Perform object detection
results = model(image)
 
# Display the image with detected objects
results.show()
 
# Get detected labels and confidences
labels = results.names  # Class names
confidences = results.xywh[0][:, 4]  # Detection confidences
predicted_classes = [labels[int(c)] for c in results.xywh[0][:, -1]]  # Predicted classes
 
# Generate description based on detected objects
description = "Detected objects: " + ", ".join(predicted_classes)
 
print(f"\n💡 Object Detection Description: {description}")