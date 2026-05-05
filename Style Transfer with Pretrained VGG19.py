import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications import vgg19

# -----------------------------
# Load content and style images
# -----------------------------
content_image_path = 'path_to_content_image.jpg'  # Replace with your content image
style_image_path = 'path_to_style_image.jpg'      # Replace with your style image

def load_and_process_image(img_path, max_dim=512):
    img = kp_image.load_img(img_path)
    img = kp_image.img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

content_array = load_and_process_image(content_image_path)
style_array = load_and_process_image(style_image_path)

# -----------------------------
# VGG19 model for feature extraction
# -----------------------------
vgg = vgg19.VGG19(weights='imagenet', include_top=False)
vgg.trainable = False

# Layers for content and style
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
    'block5_conv1'
]

def get_model_outputs(model, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.models.Model(inputs=model.input, outputs=outputs)

content_model = get_model_outputs(vgg, content_layers)
style_model = get_model_outputs(vgg, style_layers)

# -----------------------------
# Gram matrix for style
# -----------------------------
def gram_matrix(tensor):
    x = tf.transpose(tensor, perm=[0, 3, 1, 2])
    features = tf.reshape(x, [x.shape[0], x.shape[1], -1])
    gram = tf.matmul(features, features, transpose_b=True)
    return gram / tf.cast(x.shape[2] * x.shape[3], tf.float32)

# -----------------------------
# Loss functions
# -----------------------------
def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def compute_style_loss(base_style, target):
    return tf.reduce_mean(tf.square(gram_matrix(base_style) - gram_matrix(target)))

# -----------------------------
# Style transfer step
# -----------------------------
generated_image = tf.Variable(content_array, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=5.0)

content_weight = 1e3
style_weight = 1e-2
num_steps = 1000

@tf.function()
def train_step():
    with tf.GradientTape() as tape:
        # Extract features
        gen_content_features = content_model(generated_image)
        gen_style_features = style_model(generated_image)

        content_features = content_model(content_array)
        style_features = style_model(style_array)

        # Compute losses
        c_loss = tf.add_n([compute_content_loss(cf, gcf) 
                           for cf, gcf in zip(content_features, gen_content_features)])
        s_loss = tf.add_n([compute_style_loss(sf, gsf) 
                           for sf, gsf in zip(style_features, gen_style_features)])
        total_loss = content_weight * c_loss + style_weight * s_loss

    grads = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255.0 - 103.939))
    return total_loss

# -----------------------------
# Optimization loop
# -----------------------------
for i in range(num_steps):
    loss = train_step()
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.numpy():.2f}")

# -----------------------------
# Deprocess and display image
# -----------------------------
def deprocess_image(img):
    img = img.numpy().squeeze()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

final_img = deprocess_image(generated_image)
plt.imshow(final_img)
plt.axis('off')
plt.title("Neural Style Transfer Result")
plt.show()