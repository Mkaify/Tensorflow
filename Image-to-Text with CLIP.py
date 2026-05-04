from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
 
# Load an image (e.g., sample image of a dog)
image_path = "https://upload.wikimedia.org/wikipedia/commons/a/a2/American_Staffordshire_Terrier_600.jpg"
image = Image.open(image_path)
 
# Sample text descriptions to compare
text_inputs = ["a dog playing", "a cat on a bed", "a person playing soccer", "a computer programming book"]
 
# Preprocess image and text inputs
inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
 
# Get image-text similarity scores
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Convert to probabilities
 
# Get the most similar text description for the image
best_match_idx = torch.argmax(probs)
best_match = text_inputs[best_match_idx]
 
# Display results
print(f"🖼️ Image: {image_path}")
print(f"\n💬 Generated Text Description: {best_match}")