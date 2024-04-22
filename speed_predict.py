import cv2
import numpy as np
from tensorflow import keras

# Load the video file
cap = cv2.VideoCapture('speedApp/data/train.mp4')

# Load the training data (speed values)
with open('speedApp/data/train.txt', 'r') as f:
    speeds = [float(line.strip()) for line in f.readlines()]

# Create a list to store the video frames
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to grayscale and resize it to a fixed size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    frames.append(gray)

# Convert the lists to numpy arrays
frames = np.array(frames)
speeds = np.array(speeds)

# Normalize the pixel values to be between 0 and 1
frames = frames / 255.0

# Split the data into training and testing sets
train_frames, test_frames = frames[:800], frames[800:]
train_speeds, test_speeds = speeds[:800], speeds[800:]

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_frames, train_speeds, epochs=10, batch_size=32, validation_data=(test_frames, test_speeds))

# Evaluate the model
mse = model.evaluate(test_frames, test_speeds)
print(f'MSE: {mse}')