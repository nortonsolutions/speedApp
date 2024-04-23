import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Configurable variables
modelFile = "model.keras"
learning_rate = 0.025
epochs = 10
batch_size = 32
neurons = 32
testmode = True
test_filename = "speedApp/data/test.mp4"
training_video = "speedApp/data/train.mp4"
training_speeds = "speedApp/data/train.txt"
conv3d = False

def load_data(video_file, speeds_file):
    # Load the video file
    cap = cv2.VideoCapture(video_file)

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

    # Normalize the pixel values to be between 0 and 1
    frames = frames / 255.0

    if speeds_file is None:
        speeds = None
    else:
        # Load the speed values
        with open(speeds_file, 'r') as f:
            speeds = [float(line.strip()) for line in f.readlines()]
        speeds = np.array(speeds)

    return frames, speeds

def create_model(conv3d=False):
    if conv3d:
        model = keras.Sequential([
            keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(None, 64, 64, 1)),
            keras.layers.MaxPooling3D((2, 2, 2)),
            keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            keras.layers.MaxPooling3D((2, 2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model

def train_model(model, frames, speeds):
    if testmode:
        frames = frames[:128]
        speeds = speeds[:128]
        epochs = 1
    else:
        epochs = epochs

    model.fit(frames, speeds, epochs=epochs, batch_size=batch_size)

    return model

def predict_speeds(model, frames):
    if testmode:
        frames = frames[:128]

    predicted_speeds = model.predict(frames)

    return predicted_speeds

# Load training data
frames, speeds = load_data(training_video, training_speeds)

# Create and train the model
model = create_model(conv3d=conv3d)
model = train_model(model, frames, speeds)

# Save the model
model.save(modelFile)

# Load testing data
test_frames, _ = load_data(test_filename, None)

# Predict speeds
predicted_speeds = predict_speeds(model, test_frames)

# Generate histogram of predicted speeds
plt.hist(predicted_speeds, bins=50)
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Speeds')
plt.show()