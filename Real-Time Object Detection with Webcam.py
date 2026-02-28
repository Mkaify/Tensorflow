import tensorflow as tf
import cv2
import numpy as np


# ===============================
# Paths
# ===============================
MODEL_PATH = r"D:\Tensorflow\ssd_realtime"
LABELS_PATH = r"D:\Tensorflow\ssd_realtime\label_map.txt"


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
# Webcam
# ===============================
cap = cv2.VideoCapture(0)

print("Press Q to Exit")


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (320, 320)) / 255.0
    tensor = tensor[tf.newaxis, ...]


    # Detection
    detections = detector(tensor)
    detections = {k: v.numpy() for k, v in detections.items()}


    h, w, _ = frame.shape


    # Draw Boxes
    for i in range(len(detections["detection_scores"])):

        score = detections["detection_scores"][i]

        if score < 0.5:
            continue


        box = detections["detection_boxes"][i]
        class_id = int(detections["detection_classes"][i])

        label = labels[class_id]


        ymin, xmin, ymax, xmax = box

        left = int(xmin * w)
        right = int(xmax * w)
        top = int(ymin * h)
        bottom = int(ymax * h)


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    cv2.imshow("Real-Time Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()