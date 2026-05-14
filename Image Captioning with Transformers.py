from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# -----------------------------
# Load BLIP model and processor
# -----------------------------
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# -----------------------------
# Batch of images (URLs or local paths)
# -----------------------------
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/e/ec/Paris_Night.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/a2/American_Staffordshire_Terrier_600.jpg"
]

# -----------------------------
# Process and caption images
# -----------------------------
for idx, url in enumerate(image_urls):
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Preprocess for BLIP
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Display
    image.show(title=f"Image {idx+1}")
    print(f"\n🖼️ Image {idx+1} URL: {url}")
    print("📝 Generated Caption:", caption)