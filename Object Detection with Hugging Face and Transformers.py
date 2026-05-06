from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
 
# Load the pretrained DETR model and processor from Hugging Face
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)
 
# Load an image (URL or local path)
image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"  # Example image
image = Image.open(requests.get(image_url, stream=True).raw)
 
# Preprocess the image
inputs = processor(images=image, return_tensors="pt")
 
# Perform object detection
outputs = model(**inputs)
 
# Get the predicted boxes and labels
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
 
# Draw the bounding boxes on the image
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")
 
# Display the image with object detections
image.show()