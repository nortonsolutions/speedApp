# @author Norton 2023 (AI/Machine Learning capstone project - Speed and Lane Detection)
# Objective: Use Keras to train a model which can identify roadsigns in a video.
#
### speedTest - speedTest.py
### Author: Dave Norton and All the Others/AI Helping (Thank you)! 
# 
# version history:
# 
# 0.001 - feb 2024: initial version
# 0.002 - feb 2024: django WSGI app, more reporting
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
# - display_video: open the video and play it
# - load_testing_data: load the testing data and return the frames
# - load_training_data: load the training data
# - check_if_model_exists: check if the model exists
# - test_single_image: test a single image
# - process_new_video: process a new video file, editing the video in place
#
#
import os
import configparser

# import tensorflow as tf
import numpy as np
from keras import layers, losses, optimizers, metrics, utils, Sequential, models
from keras.models import load_model
# from src.backend.common import keras_tensor
from keras.callbacks import EarlyStopping

from speedApp.speed_cv import load_data, display_video, publish_predictions
from celery import shared_task

# config = configparser.ConfigParser()
# config.read('config.ini')
# example config read:
# print(config['DEFAULT']['learning_rate']) # -> "0.023"

# Configurable variables
modelFile = "model.keras"
learning_rate = 0.025
epochs = 2
batch_size = 32
neurons = 32

testmode = True
testmode_num_frames = 128
resized_frame_length = 32

test_filename = "speedApp/data/test.mp4"
training_video = "speedApp/data/train.mp4"
training_speeds = "speedApp/data/train.txt"

conv3d = False

predictionsFilename = "predictions.txt"

def create_model_last():

    model = models.Sequential(
        [
            layers.Input(batch_size=batch_size, 
                        shape=(1, 320, 240, 3)),
            
        ]
    )
    

    model.add( layers.TimeDistributed(
            layers.Conv2D(
                neurons, (18, 18), strides=(12, 12), activation='relu', padding='same')
    ))
    
    model.add( layers.BatchNormalization())
        
    # MaxPooling
    model.add( layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
    ))
    
    # model.add( layers.TimeDistributed(
    #         layers.Dense(neurons, activation='relu')
    # ))  
    
    # flatten each frame
    model.add( layers.TimeDistributed(
            layers.Flatten() 
    ))   


    
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape before Bidirectional = " + str(currentShape))
    
    model.add( layers.Bidirectional(
        layers.LSTM(neurons*2, return_sequences=True, recurrent_activation='sigmoid', activation='tanh')
            # stateful=True, input_shape=(1, -1)),
        # merge_mode='ave'
        )
    )

    model.add(layers.TimeDistributed(layers.Dropout(0.4)))

    # model.add( layers.TimeDistributed(
    #         layers.Dense(neurons*4, activation='relu')
    # ))
    
    model.add( layers.TimeDistributed(
            layers.Dense(neurons, activation='relu')
    ))  


    # flatten each frame
    model.add( layers.TimeDistributed(
            layers.Flatten() 
    ))   

    # model.add( layers.TimeDistributed(
    #         layers.Dense(neurons, activation='sigmoid')
    # ))
    
    
    # model.add( layers.Bidirectional(
    #     layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
    #     stateful=True, batch_size=batch_size, batch_input_shape=(batch_size, -1),
    #     use_bias=True
    # )))
    
    # s
    #     )
    # )
    
    model.add( 
        layers.Dense(1)
    )
    
    print("before lambda")
    model.summary()
    # #   remove the second dimension with squeeze
    # model.add(
    #     layers.TimeDistributed(
    #         layers.Lambda(lambda x: backend.squeeze(x, axis=1))
    #         # Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD))
    #     )
    # )

    print("after lambda")
    model.summary()
    # print the current shape of the model
    currentShape = model.output_shape
    print("model.output_shape final = " + str(currentShape))
    
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, weight_decay=learning_rate/epochs)
    
    # SGD optimizer:
    # optimizer = optimizers.SGD(
    #     learning_rate=learning_rate, momentum=0.9, nesterov=True, weight_decay=learning_rate/epochs)
    
    loss = losses.MeanSquaredError()
    metric = [metrics.MeanSquaredError()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    # model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    model.build()

    model.summary()
    return model

def pad_batch(data, batch_size):
    # make sure data has a length that is a multiple of batch_size
    if len(data) % batch_size == 0:
        return data
    return np.concatenate((data, data[:batch_size - (len(data) % batch_size)]), axis=0)   

def create_model(conv3d=False):
    if conv3d:
        model = Sequential([
            layers.Conv3D(neurons, (3, 3, 3), activation='relu', input_shape=(None, resized_frame_length, resized_frame_length, 1)),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(neurons*2, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Flatten(),
            layers.Dense(neurons*2, activation='relu'),
            layers.Dense(1)
        ])
    else:
        return create_model_last()

        # model = Sequential([
        #     layers.Conv2D(neurons, (3, 3), activation='relu', input_shape=(320, 240, 1)),
        #     layers.MaxPooling2D((2, 2)),
        #     layers.Conv2D(neurons*2, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),
        #     layers.Flatten(),
        #     layers.Dense(neurons*2, activation='relu'),
        #     layers.Dense(1)
        # ])

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model

def shape_status(model, identifier = "model.output_shape"):
    print("shapeStatus[" + identifier + "] =" + str(model.output_shape))

def save_model(model: Sequential, modelFile: str = modelFile):
    """ the expectation is to receive a trained, compiled, and built model for saving """
    try:
        model.save(modelFile)
        # model.save_weights(modelFile + ".h5")
        return

    except Exception as e:
        raise e

def load_model_from_disk(modelFile = modelFile) -> Sequential:
    """ load the model from the disk and return the model, ready to predict """
    try:
        model: Sequential = load_model(modelFile, compile=False, safe_mode=False)
        # model.load_weights(modelFile + ".h5")
        
        # config.enable_unsafe_deserialization() - not needed
        # with open(modelFile,'rb') as f:
        #     loaded_model = pickle.load(f)
        # backend.deserialize_keras_object(loaded_model, safe_mode=False)
        return model
    
    except Exception as e:
        raise e

def train_model(model, frames, speeds, epochs=epochs, batch_size=batch_size):

    frames = pad_batch(frames, batch_size)
    speeds = pad_batch(speeds, batch_size)

    try:
        # Define the callback 
        early_stopping = EarlyStopping(
            monitor='loss',  # Metric to monitor
            patience=5,  # Number of epochs to wait before stopping
            min_delta=0.001,  # Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights=True  # Restore model weights from the epoch with the best value
        )

        model.fit(frames, speeds, epochs=epochs, batch_size=batch_size, 
                verbose=2, shuffle=False, validation_split=0.0, callbacks=[early_stopping])
        # , steps_per_epoch=frames.shape[0]//batch_size)
    except Exception as e:
        raise e

    print("Training model... frames.shape:", frames.shape, "speeds.shape:", speeds.shape)
    if testmode is True:
        frames = frames[:testmode_num_frames]
        speeds = speeds[:testmode_num_frames]
        epochs = 1
    else:
        epochs = epochs
    print("Training model II ... frames.shape:", frames.shape, "speeds.shape:", speeds.shape)
    model.fit(frames, speeds, epochs=epochs, batch_size=batch_size)
    return model

def predict_speeds(model, frames):

    frames = pad_batch(frames, batch_size)

    print("Predicting speeds...", frames.shape)
    if testmode:
        frames = frames[:testmode_num_frames]

    predicted_speeds = model.predict(frames)
    # output_from_predict = trainedModel.predict(testFrames, verbose=2, batch_size=batch_size, steps=testFrames.shape[0]//batch_size)

    print("predicted_speeds.shape = ", predicted_speeds.shape)
    predicted_speeds = np.squeeze(predicted_speeds, axis=1)
    np.savetxt(predictionsFilename, predicted_speeds, fmt='%d')


    return predicted_speeds

def check_if_model_exists(modelFile = modelFile):
    # If the model file exists, load the model
    print("modelFile = ", modelFile)
    
    # print the current project working directory
    # print("os.getcwd() = ", os.getcwd())
    if os.path.exists(modelFile):
        print("WARNING: Loading model from pre-existing file.  Delete the file to retrain the model from scratch.")
        loaded_model: Sequential = None
        try:
            
            # loaded_model.build()
            loaded_model = load_model_from_disk(modelFile)
            
        except Exception as e:
            raise e
        
        print("loaded_model")
        loaded_model.summary()
        return True, loaded_model
    else:
        return False, None

def main():

    model_exists, model = check_if_model_exists(modelFile)

    if (model_exists is False):
    
        # Load training data
        frames, speeds = load_data(training_video, training_speeds, testmode=testmode, testmode_num_frames=testmode_num_frames, batch_size=batch_size)

        # Create and train the model
        model = create_model(conv3d=False)

        model.summary()
        try:
            model = train_model(model, frames, speeds, epochs=epochs, batch_size=batch_size)
        except Exception as e:
            print("Exception in training model: ", e)
            raise e
        
        save_model(model, modelFile)

    model.summary()

    # # Save the model
    # model.save(modelFile)

    # Load testing data
    test_frames, _ = load_data(test_filename, None, testmode=testmode, testmode_num_frames=testmode_num_frames, batch_size=batch_size)

    # Predict speeds
    predicted_speeds = predict_speeds(model, test_frames)

    print("predicted_speeds.shape = ", predicted_speeds.shape)

    filename = publish_predictions(predicted_speeds, test_frames)
    display_video(filename)
    
    # # Generate histogram of predicted speeds
    # plt.hist(predicted_speeds, bins=50)
    # plt.xlabel('Speed')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Predicted Speeds')
    # plt.show()

    # Save the predicted speeds to a file

# processNewVideo will process a new video file, editing the video in place
@shared_task
def process_new_video(filename):

    model_exists, model = check_if_model_exists(modelFile)

    if (model_exists is False):

        # Load training data
        frames, speeds = load_data(training_video, training_speeds, testmode=testmode, testmode_num_frames=testmode_num_frames, batch_size=batch_size)

        # Create and train the model
        model = create_model(conv3d=False)

        model.summary()
        model = train_model(model, frames, speeds, epochs=epochs, batch_size=batch_size)

    model.summary()

    # # Save the model
    # model.save(modelFile)

    # Load testing data
    test_frames, _ = load_data(test_filename, None, testmode=testmode, testmode_num_frames=testmode_num_frames, batch_size=batch_size)

    # Predict speeds
    predicted_speeds = predict_speeds(model, test_frames)

    print("predicted_speeds.shape = ", predicted_speeds.shape)

    filename = publish_predictions(predicted_speeds, test_frames, filename)
    display_video(filename)
    
    return "Success"

if __name__ == "__main__":
    main()

# TODO: broken
def test_single_image():
    frames = ld_testing_data()
    
    # Remove the second dimension
    frames = np.squeeze(frames, axis=1)
    print("frames.shape = ", frames.shape)
    
    image = frames[400]
    superImposedImage = superimpose_image(18.01, image)
    # display_image("superimposed", superImposedImage)
    imageFinal = image_with_lanes(superImposedImage)
    # display_image("final", imageFinal)
    return imageFinal