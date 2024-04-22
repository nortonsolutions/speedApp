import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# Load the video file
cap = cv2.VideoCapture('speedApp/data/train.mp4')

# Load the speed values from the text file
# with open('speedApp/data/train.txt', 'r') as f:
#     speeds = [float(line.strip()) for line in f.readlines()]
# speeds = np.array(speeds)

speeds = np.loadtxt('speedApp/data/train.txt')

# Create a list to store the frames
frames = []

# Read the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to grayscale and resize it to a fixed size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    frames.append(gray)

# Convert the frames to a numpy array
frames = np.array(frames)

# Split the data into training and testing sets
train_frames, test_frames, train_speeds, test_speeds = train_test_split(frames, speeds, test_size=0.2, random_state=42)

# Normalize the pixel values to be between 0 and 1
train_frames = train_frames / 255.0
test_frames = test_frames / 255.0

# Another way to do the above:
# Normalize the pixel values to be between 0 and 1
# frames = frames / 255.0

# # Split the data into training and testing sets
# train_frames, test_frames = frames[:800], frames[800:]
# train_speeds, test_speeds = speeds[:800], speeds[800:]

print(frames.shape)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(frames.shape[1], frames.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_frames, train_speeds, epochs=2, batch_size=32, validation_data=(test_frames, test_speeds))

# Evaluate the model
mse = model.evaluate(test_frames, test_speeds)
print(f'MSE: {mse}')