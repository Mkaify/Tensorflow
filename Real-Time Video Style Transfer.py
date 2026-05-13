import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications.vgg19 import preprocess_input

# -----------------------------
# Paths
# -----------------------------
style_image_path = 'starry_night.jpg'       # Style image (painting)
content_video_path = 'input_video.mp4'      # Input video
output_video_path = 'output_video.mp4'      # Stylized video output

# -----------------------------
# Load style image
# -----------------------------
style_image = kp_image.load_img(style_image_path)
style_image = kp_image.img_to_array(style_image)
style_image = np.expand_dims(style_image, axis=0)
style_image = preprocess_input(style_image)

# -----------------------------
# Load VGG19 for style/content features
# -----------------------------
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']

def get_model():
    """Build a model that outputs style and content layers"""
    outputs = [vgg.get_layer(layer).output for layer in style_layers + content_layers]
    return tf.keras.Model(vgg.input, outputs)

model = get_model()

# -----------------------------
# Placeholder style transfer function
# -----------------------------
def style_transfer(frame, model, style_image):
    """
    Apply style transfer to a single frame.
    This is a simplified placeholder. In practice, you compute
    content loss, style loss, and optimize using gradients.
    """
    content_image = np.expand_dims(frame, axis=0)
    content_image = preprocess_input(content_image)

    # TODO: Implement neural style transfer:
    # 1. Extract features from content_image and style_image
    # 2. Compute content & style loss
    # 3. Optimize generated frame (gradient descent)
    
    # For demo purposes, return original frame
    return frame

# -----------------------------
# Open input video
# -----------------------------
cap = cv2.VideoCapture(content_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# -----------------------------
# Process video frames
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply style transfer to the current frame
    stylized_frame = style_transfer(frame, model, style_image)
    
    # Write the processed frame
    out.write(stylized_frame)
    
    # Optional: display the frame
    cv2.imshow('Stylized Video', stylized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Stylized video saved as", output_video_path)