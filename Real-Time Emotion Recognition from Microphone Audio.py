# ⚠️ Requires: sounddevice, librosa, and a working microphone.

import tensorflow as tf
import numpy as np
import sounddevice as sd
import librosa
 
# Parameters
sr = 16000                  # Sampling rate
duration = 2.0              # Record duration in seconds
n_mfcc = 13                 # Number of MFCCs
max_len = 100               # Max time steps for padding
 
# Record audio from microphone
print("🎙️ Speak now...")
recording = sd.rec(int(sr * duration), samplerate=sr, channels=1)
sd.wait()
audio = np.squeeze(recording)
 
# Extract MFCCs
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
mfcc = mfcc.T                                                  # Transpose to (time, features)
 
# Pad/truncate to fixed length
mfcc_padded = tf.keras.preprocessing.sequence.pad_sequences([mfcc], maxlen=max_len, padding='post', dtype='float32')
 
# Dummy model (replace with pre-trained emotion model)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len, n_mfcc)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')             # 3 emotions: happy, sad, angry
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.set_weights([np.random.randn(*w.shape) * 0.05 for w in model.get_weights()])  # Simulated weights
 
# Predict emotion
pred = model.predict(mfcc_padded)[0]
emotion_index = np.argmax(pred)
emotions = ['Happy', 'Sad', 'Angry']
print("🧠 Detected Emotion:", emotions[emotion_index])