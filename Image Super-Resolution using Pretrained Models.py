import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Load pretrained super-resolution model
# -----------------------------
# Replace with the path to your .h5 model file (e.g., ESRGAN or FSRCNN)
model = load_model("path_to_super_resolution_model.h5")  

# -----------------------------
# Read and preprocess low-resolution image
# -----------------------------
low_res_path = "low_resolution_image.jpg"  # Replace with your low-res image
image = cv2.imread(low_res_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# Resize image for model input if necessary (depends on model requirements)
# Here, we simply upscale by factor of 2 for demonstration
image_resized = image_pil.resize((image_pil.width * 2, image_pil.height * 2))

# Convert to numpy array and normalize
image_array = np.array(image_resized) / 255.0
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# -----------------------------
# Perform super-resolution
# -----------------------------
high_res_image = model.predict(image_array)

# Postprocess: remove batch dimension, denormalize, clip values
high_res_image = np.squeeze(high_res_image) * 255.0
high_res_image = np.clip(high_res_image, 0, 255).astype("uint8")

# -----------------------------
# Display and save high-resolution image
# -----------------------------
high_res_pil = Image.fromarray(high_res_image)
high_res_pil.show()
high_res_pil.save("high_resolution_image.jpg")

print("✅ Super-resolution complete! Image saved as high_resolution_image.jpg")