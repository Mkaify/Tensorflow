from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# -----------------------------
# Load a text-to-image model (Stable Diffusion as a proxy for DALL·E)
# -----------------------------
model_name = "runwayml/stable-diffusion-v1-5"  # High-quality SD model
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU if available

# -----------------------------
# Input prompt
# -----------------------------
prompt = "A futuristic city skyline at sunset, with flying cars and neon lights"

# -----------------------------
# Generate image
# -----------------------------
generated_image = pipe(prompt, guidance_scale=7.5).images[0]

# -----------------------------
# Display the image
# -----------------------------
generated_image.show()

# -----------------------------
# Save the generated image
# -----------------------------
generated_image.save("generated_image.png")
print("✅ Image saved as generated_image.png")