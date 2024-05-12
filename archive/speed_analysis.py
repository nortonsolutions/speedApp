# @author Norton 2023 (AI/Machine Learning capstone project - Speed and Lane Detection)
# Objective: Use Keras to train a model which can identify roadsigns in a video.
#
### speedTest - speedTest.py
### Author: Dave Norton and All the Others/AI Helping (Thank you)! 
# 
# version history:
# 
# 0.001 - feb 2024: initial version
#
# -----
#
# This file contains the functions for the machine learning part of the speedTest project.
# The functions are used to train the model and test the model on the test data.
# The functions are used in the speedTest.py file.
# The functions are:
# - create_model: create the model
# - train_model: train the model on the training data
# - test_model: test the model on the test data
# - publish_predictions: superimpose the predictions on the testFrames, add the lanelines, and display the resulting video
# - displayVideo: open the video and play it
# - ld_testing_data: load the testing data and return the frames
# - load_training_data: load the training data
# - check_if_model_exists: check if the model exists
# - test_single_image: test a single image
# - processNewVideo: process a new video file, editing the video in place
#
#
import numpy as np
import cv2
import os
import configparser
# import keras.backend as K
# from asgiref.sync import async_to_sync, sync_to_async
from tensorflow import keras
from tensorflow.keras import backend as K
from celery import shared_task

config = configparser.ConfigParser()
config.read('config.ini')

# example config read:
# print(config['DEFAULT']['learning_rate']) # -> "0.023"

import matplotlib.pyplot as plt

# Configurable variables
modelFile = "model.keras"
learning_rate = 0.025
# The learning_rate is the rate at which the model learns.
# The learning_rate is multiplied by the gradient to determine the amount to adjust the weights.
# The learning_rate is a hyperparameter that can be tuned to improve the model, like how much
# room to wiggle the weights when adjusting them on each pass.

epochs = 10
batch_size = 32
neurons = 32
testmode = True
test_filename = "speedApp/data/test.mp4"
training_video = "speedApp/data/train.mp4"
training_speeds = "speedApp/data/train.txt"

predictionsFilename = "predictions.txt"

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
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        frames.append(frame)

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
        create_model_last()
        # model = keras.Sequential([
        #     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        #     keras.layers.MaxPooling2D((2, 2)),
        #     keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D((2, 2)),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(64, activation='relu'),
        #     keras.layers.Dense(1)
        # ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model


def create_model_last():

    model = keras.models.Sequential(
        [
            keras.Input(batch_size=batch_size, 
                        shape=(1, 320, 240, 3)),
            
        ]
    )
    

    model.add( keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                neurons, (18, 18), strides=(12, 12), activation='relu', padding='same')
    ))
    
    model.add(keras.layers.BatchNormalization())
        
    # MaxPooling
    model.add( keras.layers.TimeDistributed(
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
    ))
    
    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.Dense(neurons, activation='relu')
    # ))  
    
    # flatten each frame
    model.add( keras.layers.TimeDistributed(
            keras.layers.Flatten() 
    ))   


    
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape before Bidirectional = " + str(currentShape))
    
    model.add( keras.layers.Bidirectional(
        keras.layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
            stateful=True, input_shape=(1, -1)),
        merge_mode='ave'
        )
    )

    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))

    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.Dense(neurons*4, activation='relu')
    # ))
    
    model.add( keras.layers.TimeDistributed(
            keras.layers.Dense(neurons, activation='relu')
    ))  


    # flatten each frame
    model.add( keras.layers.TimeDistributed(
            keras.layers.Flatten() 
    ))   

    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.Dense(neurons, activation='sigmoid')
    # ))
    
    
    # model.add( keras.layers.Bidirectional(
    #     keras.layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
    #     stateful=True, batch_size=batch_size, batch_input_shape=(batch_size, -1),
    #     use_bias=True
    # )))
    
    # s
    #     )
    # )
    
    model.add( 
        keras.layers.Dense(1)
    )
    
    print("before lambda")
    model.summary()
    # #   remove the second dimension with squeeze
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=1))
            # Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD))
        )
    )

    print("after lambda")
    model.summary()
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape final = " + str(currentShape))
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=learning_rate/epochs)
    
    # SGD optimizer:
    # optimizer = keras.optimizers.SGD(
    #     learning_rate=learning_rate, momentum=0.9, nesterov=True, weight_decay=learning_rate/epochs)
    
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    model.build()

    model.summary()
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

def shapeStatus(model, identifier = "model.output_shape"):
    print("shapeStatus[" + identifier + "] =" + str(model.output_shape))

def create_modelOld():

    model = keras.models.Sequential(
        [
            keras.Input(batch_size=batch_size, 
                        batch_shape=(batch_size, 1, 320, 240, 3)),
        ]
    )
    
    # All things being equal, why not Dropout right at the beginning?
    # model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))
    
    # model.add(keras.layers.BatchNormalization())
    model.add( keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                neurons, (18, 18), strides=(12, 12), activation='relu')
    ))
    
    
    model.add( keras.layers.TimeDistributed(
            keras.layers.Dense(neurons, activation='relu')
    ))  
    
    model.add( keras.layers.Bidirectional(
        keras.layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
        stateful=True, batch_size=batch_size, batch_input_shape=(batch_size, -1),    
    )))
    
    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.
    # ))
    
    # model.add(keras.layers.BatchNormalization())
    # print the current shape of the model
    # currentShape = model.output_shape
    # print("model.output_shape after Conv2D part 2 = " + str(currentShape))    
    
    model.add( keras.layers.TimeDistributed(
            keras.layers.Dense(neurons, activation='relu')
    ))  

    # flatten each frame
    model.add( keras.layers.TimeDistributed(
            keras.layers.Flatten() 
    ))   

    model.add( keras.layers.TimeDistributed(
            keras.layers.Dense(neurons, activation='sigmoid')
    ))
    
    # remove the second dimension with squeeze
    # model.add(
    #     keras.layers.TimeDistributed(
    #         keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=1))
    #         # Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD))
    #     )
    # )
    
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape before Bidirectional = " + str(currentShape))
    
    model.add( keras.layers.Bidirectional(
    keras.layers.LSTM(neurons*4, return_sequences=False, recurrent_activation='sigmoid', activation='tanh',
        stateful=True, batch_size=batch_size, batch_input_shape=(batch_size, -1),    
    )))

    # model.add( keras.layers.Bidirectional(
    #     keras.layers.LSTM(neurons*4, return_sequences=False, recurrent_activation='sigmoid', activation='tanh',
    #     stateful=True, batch_size=batch_size, batch_input_shape=(batch_size, -1),
    # )))
    
    # model.add( keras.layers.Bidirectional(
    #     keras.layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
    #     stateful=False, # , unroll=False, use_bias=True,
    #     batch_size=batch_size, batch_input_shape=(batch_size, -1),
    # )))

    # # print the current shape of the model
    # currentShape = model.output_shape
    # print("model.output_shape after Bidirectional = " + str(currentShape))
    
    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.Dense(neurons*8, activation='sigmoid')
    # ))  

    # # model.add( # keras.layers.TimeDistributed(
    # #         keras.layers.Dense(neurons*4)
    # #     # )
    # # )  
    
    # model.add( keras.layers.TimeDistributed(
    #         keras.layers.Dense(neurons*2, activation='sigmoid')
    # )) 
    
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(neurons)
        )
    )
    
    model.add( keras.layers.TimeDistributed(
        keras.layers.Dense(1)
    ))

    # # print the current shape of the model
    # currentShape = model.output_shape
    # print("model.output_shape before Flatten layer = " + str(currentShape))
    
    # model.add(
    #     keras.layers.Flatten(batch_input_shape=(batch_size, -1)),
    # )

    # model.add(
    #     keras.layers.Flatten(input_shape=(batch_size, -1))  
    # )
    
    # print the current shape of the model
    # currentShape = model.output_shape
    # print("model.output_shape after Flatten = " + str(currentShape))
    
    # model.add(
    #     keras.layers.Dense(1)
    # )

    # model.add(
    #     keras.layers.Lambda(lambda x: K.permute_dimensions(x, [1,4]))
    #     # Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD))
    # )
    
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape final = " + str(currentShape))
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, decay=learning_rate/epochs)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build()

    model.summary()
    return model

# train_model will train the model on the training data, save the model, and return the model
def train_model(model, frames, train):

    # Convert frames and train from numpy arrays to keras pydatasets
    model.fit(frames, train, epochs=epochs, batch_size=batch_size, shuffle=False, validation_split=0.2, verbose=2)

    model.save(modelFile)
    return

# test_model will test the model on the test data and create a prediction file
def test_model(testFrames):

    print("testFrames.shape = ", testFrames.shape)    
    trainedModel: keras.models.Sequential = keras.models.load_model(modelFile)
    
    # testFrames = np.squeeze(testFrames, axis=1)
    output_from_predict = trainedModel.predict(testFrames, verbose=2, batch_size=batch_size, steps=testFrames.shape[0]//batch_size)
                                            #  steps=testFrames.shape[0]//batch_size,
    print("output_from_predict.shape = ", output_from_predict.shape)
    
    output_from_predict = np.squeeze(output_from_predict, axis=1)
    np.savetxt(predictionsFilename, output_from_predict, fmt='%d')
    
    return output_from_predict

def predict_speeds(model, frames):
    if testmode:
        frames = frames[:128]

    predicted_speeds = model.predict(frames)

    return predicted_speeds

# Load training data
frames, speeds = load_data(training_video, training_speeds)

# Create and train the model
model = create_model(conv3d=False)
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


# publish_predictions will superimpose the predictions on the testFrames,
# add the lanelines, and display the resulting video 
def publish_predictions(predictions, testFrames):

    filename = "predictions.mp4"
    newVideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (320, 240))

    # Remove the second dimension
    testFrames = np.squeeze(testFrames, axis=1)
    print("testFrames.shape = ", testFrames.shape)
    print("predictions.shape = ", predictions.shape)
    
    for i in range(len(predictions)):
        # print("predictions[i] = ", predictions[i])
        # print("testFrames[i] = ", testFrames[i])

        superImposedImage = superimposeImage(predictions[i], testFrames[i])
        # print("superImposedImage.shape = ", superImposedImage.shape)
        # display_image("superimposed", superImposedImage)
        # superImposedImage = np.squeeze(superImposedImage, axis=1)
        imageFinal = image_with_lanes(superImposedImage)
        
        # display_image("final", imageFinal)
        newVideo.write(imageFinal)

    # Release the video
    newVideo.release()
    
    return filename

# Open the video and play it
def displayVideo(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break    

            
    cap.release()
    cv2.destroyAllWindows()
    return

def superimposeImage(prediction, image):
    superimposedImage = cv2.putText(image, str(prediction), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # print("superimposedImage.shape = ", superimposedImage.shape)
    return superimposedImage

# ld_testing_data will load the testing data and return the frames
def ld_testing_data(filename):

    # Load the video file
    cap = cv2.VideoCapture(filename)
    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 1000
    if (testmode is True):
        num_frames = 1000
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("height = ", height)
    print("width = ", width)

    # If the length of the first dimension of frames is less than a multiple of batch_size,
    # change the size of that dimension to a multiple of batch_size
    if (num_frames % batch_size != 0):
        num_frames = (num_frames // batch_size) * batch_size
        
    # Create an array to store the frames using window_nd
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading test frame", i)
        frames[i] = shrink_img(cap.read()[1])

    frames = np.expand_dims(frames, axis=1)

    # Close the video file
    cap.release()
    
    # Return the frames
    return frames[0:num_frames]

# Load the training data
def load_training_data():

    # Load the video file
    cap = cv2.VideoCapture(training_video)
    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 1000
    if (testmode is True):
        num_frames = 1000
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create an array to store the frames using window_nd

    # If the length of the first dimension of frames is less than a multiple of batch_size,
    # change the size of that dimension to a multiple of batch_size
    if (num_frames % batch_size != 0):
        num_frames = (num_frames // batch_size) * batch_size
    
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Load the train.txt file
    train = np.loadtxt(training_speeds) # , dtype='float32')

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading training frame", i)
        frames[i] = shrink_img(cap.read()[1])

    # Close the video file
    cap.release()

    print("frames.shape = ", frames.shape)
    print("train.shape = ", train.shape)

    frames = np.expand_dims(frames, axis=1)
    train = np.expand_dims(train, axis=1)

    # Return the frames
    return frames[0:num_frames], train[0:num_frames]


def check_if_model_exists(modelFile = modelFile):
    # If the model file exists, load the model
    print("modelFile = ", modelFile)
    
    # print the current project working directory
    # print("os.getcwd() = ", os.getcwd())
    if os.path.exists(modelFile):
        print("WARNING: Loading model from pre-existing file.  Delete the file to retrain the model from scratch.")
        model = keras.models.load_model(modelFile)
        model.summary()
        return True, model
    else:
        return False, None
    

def test_single_image():
    frames = ld_testing_data()
    
    # Remove the second dimension
    frames = np.squeeze(frames, axis=1)
    print("frames.shape = ", frames.shape)
    
    image = frames[400]
    superImposedImage = superimposeImage(18.01, image)
    # display_image("superimposed", superImposedImage)
    imageFinal = image_with_lanes(superImposedImage)
    # display_image("final", imageFinal)
    return imageFinal
    
# Lane finding Pipeline based on https://www.kaggle.com/code/soumya044/lane-line-detection

def main():

    modelExists, model = check_if_model_exists()

    if (modelExists is False):
        print("Model does not exist, creating model...")
        print("Loading training data...")
        frames, train = load_training_data()

        print("frames.shape = ", frames.shape)
        print("train.shape = ", train.shape)

        print("Loading model...")
        model = create_model()

        model.summary()

        print("Training model...")
        train_model(model, frames, train)
        print("Training complete...")

    model.summary()
    
    print("Loading Testing data...")
    testFrames = ld_testing_data(test_filename)

    print("testFrames.shape before test = ", testFrames.shape)
    
    print("Testing complete, creating predictions...")
    predictions = test_model(testFrames)
    
    # describe the predictions
    print("predictions.shape = ", predictions.shape)
    # print("predictions = ", predictions)

    filename = publish_predictions(predictions, testFrames)
    displayVideo(filename)
    
    return

# processNewVideo will process a new video file, editing the video in place
@shared_task
def processNewVideo(filename):
    err = "testing 123"
    
    modelExists, model = check_if_model_exists(modelFile)

    if (modelExists is False):
        return "Model does not exist, please train the model first."

    print("Loading data...")
    testFrames = ld_testing_data(filename)

    print("Testing complete, creating predictions...")
    predictions = test_model(testFrames)
    
    # describe the predictions
    print("predictions.shape = ", predictions.shape)
    # print("predictions = ", predictions)
    
    _ = publish_predictions(predictions, testFrames, filename)
    
    return err

if __name__ == "__main__":
    main()

