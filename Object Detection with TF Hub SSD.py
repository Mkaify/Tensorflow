import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Paths
# ===============================
MODEL_PATH = r"D:\Tensorflow\ssd_model"
LABELS_PATH = r"D:\Tensorflow\ssd_model\label_map.txt"
IMAGE_PATH = r"D:\Tensorflow\ssd_model\test.jpg"


# ===============================
# Load Model
# ===============================
detector = tf.saved_model.load(MODEL_PATH)


# ===============================
# Load Labels
# ===============================
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]


# ===============================
# Load Image
# ===============================
img_raw = tf.io.read_file(IMAGE_PATH)
img = tf.image.decode_jpeg(img_raw, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = img[tf.newaxis, ...]


# ===============================
# Detection
# ===============================
result = detector(img)

result = {k: v.numpy() for k, v in result.items()}


# ===============================
# Visualization
# ===============================
image_np = img[0].numpy()

plt.figure(figsize=(10, 6))
plt.imshow(image_np)

ax = plt.gca()


for i in range(len(result["detection_scores"])):

    score = result["detection_scores"][i]

    if score < 0.5:
        continue

    box = result["detection_boxes"][i]

    class_id = int(result["detection_classes"][i])

    label = labels[class_id]


    ymin, xmin, ymax, xmax = box

    h, w, _ = image_np.shape

    left = xmin * w
    right = xmax * w
    top = ymin * h
    bottom = ymax * h


    ax.add_patch(
        plt.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            edgecolor="red",
            facecolor="none",
            linewidth=2
        )
    )

    ax.text(
        left,
        top - 5,
        f"{label}: {score:.2f}",
        color="red",
        fontsize=10,
        backgroundcolor="white"
    )


plt.title("Object Detection with SSD MobileNet V2")
plt.axis("off")
plt.show()