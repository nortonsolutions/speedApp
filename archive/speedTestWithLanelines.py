# @author Norton 2023 (AI/Machine Learning capstone project - Speed and Lane Detection)
# Objective: Use Keras to train a model which can identify roadsigns in a video.
#
### speedTest - speedTest.py
### Author: Dave Norton and All the Others/AI Helping (Thank you)! 
# 
# version history:
# 
# 0.001 - feb 2024: initial version

import os
import configparser

# import tensorflow as tf
import numpy as np
from keras import layers, losses, optimizers, metrics, utils, Sequential, models
from keras.models import load_model
# from src.backend.common import keras_tensor
from keras.callbacks import EarlyStopping

import speedTestCV
from celery import shared_task

# import tensorflow as tf

config = configparser.ConfigParser()
config.read('config.ini')


# example config read:
# print(config['DEFAULT']['learning_rate']) # -> "0.023"

modelFile = "./model.keras"
learning_rate = 0.025
# The learning_rate is the rate at which the model learns.
# The learning_rate is multiplied by the gradient to determine the amount to adjust the weights.
# The learning_rate is a hyperparameter that can be tuned to improve the model, like how much
# room to wiggle the weights when adjusting them on each pass.

epochs = 2
batch_size = 32
neurons = 32
testmode = True
testFilename = "speedTestTrainingData/test.mp4"
trainingVideo = "speedTestTrainingData/train.mp4"
trainingSpeeds = "speedTestTrainingData/train.txt"

predictionsFilename = "predictions.txt"

safe_mode=False



def pad_batch(data, batch_size):
    # make sure data has a length that is a multiple of batch_size
    if len(data) % batch_size == 0:
        return data
    return np.concatenate((data, data[:batch_size - (len(data) % batch_size)]), axis=0)

def createModelB(batch_size=None) -> models.Sequential:
    model = models.Sequential()

    model.add(layers.Input(shape=(None, 240, 320, 3)))  # Variable-length sequences

    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
                              padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(filters=32, kernel_size=(3, 3),
                              padding='same', return_sequences=False))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    return model

def createModel(batch_size=None) -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Input(shape=(1, 240, 320, 3)))  

    # ConvLSTM Block
    model.add(layers.ConvLSTM2D(filters=32, kernel_size=(18, 18), strides=(12, 12), padding='same', return_sequences=True, activation='tanh'))
    model.add(layers.BatchNormalization())  # Consider for stability
    
    # Additional ConvLSTM?  
    # model.add(...)  # Potential for a second ConvLSTM2D 

    model.add(layers.Flatten()) 
    model.add(layers.Dense(128, activation='relu')) 
    model.add(layers.Dense(1)) 

    # Similar optimizer, loss, metrics as before
    optimizer = optimizers.RMSprop(learning_rate=0.001)  
    loss = losses.MeanAbsoluteError()  
    mse_metrics = [metrics.MeanSquaredError()] 
    model.compile(optimizer=optimizer, loss=loss, metrics=mse_metrics)
    # model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

def createModel_last(batch_size = None) -> models.Sequential:  # Remove default batch_size

    model = Sequential()


    model.add(layers.Input(shape=(1, 240, 320, 3)))  # No fixed batch size

    # Convolutional Layers with Batch Normalization
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten and Dense Layers
    model.add(layers.Flatten()) 
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(1))

    # Compilation
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    return model


def createModel_alt(batch_size = None) -> models.Sequential:  # Remove default batch_size
    model = models.Sequential()

    model.add(layers.Input(shape=(1, 240, 320, 3)))  # No fixed batch size


    # Variation: Convolutional block with Max Pooling 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Variation: Deeper convolutions
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())  # Prepare for dense layers

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))  # Regularization

    model.add(layers.Dense(1))

    # Optimizer, loss, and metric choices (explained below)
    optimizer = optimizers.RMSprop(learning_rate=0.001)  
    loss = losses.MeanAbsoluteError()  
    mse_metrics = [metrics.MeanSquaredError()] 

    model.compile(optimizer=optimizer, loss=loss, metrics=mse_metrics)

    return model
# Changes

# Removed Batch Size from Input: Enhances model flexibility.
# Convolutional Block: Introduced a common pattern of a convolutional layer followed by max pooling for downsampling.
# Deeper Convolutions:  Added another convolutional block for potentially learning more complex features.
# Flatten: Prepares the output of the convolutional layers for the dense layers.
# Dense with Dropout: Added regularization using Dropout.

# Optimizer, Loss, and Metric Considerations

# Optimizer: I switched to RMSprop. It's an adaptive optimizer often well-suited for computer vision tasks. However, Adam remains a solid choice as well.  
# Loss: I changed the loss function to mean absolute error (MAE). MAE is sometimes more robust to outliers than mean squared error (MSE). 
# Metrics: I kept MSE as a metric since it gives a sense of the average squared error, which can be helpful for monitoring.

# Experimentation is Key:  The best choices for your model depend heavily on the nature of your dataset and the specific problem you're trying to solve. Experiment with different layers, optimizers, losses, and hyperparameters.  
# Task-Specific Considerations:  If your output should be within a certain range or represent probabilities, you might need an activation function on your output layer (e.g., 'sigmoid' for probabilities).
# Potential for LSTM: If the temporal aspect of the input data is important, instead of GRU, consider using layers like LSTM (Long Short-Term Memory) which are designed to handle longer-term dependencies in sequential data.

# Let me know if you have a specific task in mind, and I can provide even more tailored suggestions!


def createModel_orig(batch_size: int = batch_size) -> Sequential:
    """Returns model that is uncompiled, unbuilt, untrained."""
    
    model = Sequential(
        [
            layers.Input( 
                batch_size=batch_size,
                shape=(1, 240, 320, 3),               
            )
        ]
    )
    
    model.add( 
        layers.TimeDistributed(
            layers.Conv2D(neurons, (18, 18), strides=(12, 12), activation='relu', padding='same'),
        )
    )
    
    # flatten each frame
    model.add( layers.TimeDistributed(
        layers.Flatten(),
    ))   

    # model.add( 
        # layers.Bidirectional(
            # layers.LSTM(neurons*4, return_sequences=True, recurrent_activation='sigmoid', activation='tanh',
            #     stateful=True, use_bias=False, unroll=True), 
            # name='bd_lstm_1',
            # merge_mode='ave',
            # input_shape=(None, None),
    # )
    
    model.add(
        layers.GRU(neurons * 2, activation='tanh', return_sequences=True)
    )
    
    model.add( layers.TimeDistributed(
        layers.Dense(neurons, activation='relu'),
    ))  
    
    model.add(
        layers.TimeDistributed(
            layers.Dense(neurons)
        )
    )
    
    model.add( 
        layers.Dense(1)
    )

    # #   remove the second dimension with squeeze
    # model.add(
    #     layers.TimeDistributed(
    #         layers.Lambda(lambda x: backend.common.keras_tensor.squeeze(x, axis=1)),
    #         # Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD))
    #     )
    # )
    # print the current shape of the model
    # currentShape = model.output_shape
    # print("model.output_shape final = " + str(currentShape))

    
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, weight_decay=learning_rate/epochs)
    
    # SGD optimizer:
    # optimizer = optimizers.SGD(
    #     learning_rate=learning_rate, momentum=0.9, nesterov=True, weight_decay=learning_rate/epochs)
    
    loss = losses.MeanSquaredError()
    mse_metrics = [metrics.MeanSquaredError()]
    model.compile(optimizer=optimizer, loss=loss, metrics=mse_metrics)

    return model

def shapeStatus(model, identifier = "model.output_shape"):
    print("shapeStatus[" + identifier + "] =" + str(model.output_shape))

def saveModel(model: Sequential, modelFile: str = modelFile):
    """ the expectation is to receive a trained, compiled, and built model for saving """
    try:
        model.save(modelFile)
        # model.save_weights(modelFile + ".h5")
        return

    except Exception as e:
        raise e

def loadModelFromDisk(modelFile = modelFile) -> Sequential:
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

def trainModel(model, frames, train):
    """trainModel receives compiled model to train on the training data.  Return the model"""
    frames = pad_batch(frames, batch_size)
    train = pad_batch(train, batch_size)
    
    print("frames.shape = ", frames.shape)
    print("train.shape = ", train.shape)
    try:
        # Define the callback 
        early_stopping = EarlyStopping(
            monitor='loss',  # Metric to monitor
            patience=5,  # Number of epochs to wait before stopping
            min_delta=0.001,  # Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights=True  # Restore model weights from the epoch with the best value
        )

        model.fit(frames, train, epochs=epochs, batch_size=batch_size, 
                verbose=2, shuffle=False, validation_split=0.0, callbacks=[early_stopping])
        # , steps_per_epoch=frames.shape[0]//batch_size)
    except Exception as e:
        raise e
    
    # utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=False,
    #     show_dtype=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False, 
    #     dpi=96,
    #     layer_range=None,
    #     show_layer_activations=True,
    #     show_trainable=False
    # )
    return model

def testmodelA(testFrames, trainedModel: Sequential):
    testFrames = pad_batch(testFrames, batch_size)  # Ensure consistent padding

    outputFromPredict = trainedModel.predict(testFrames, batch_size=batch_size)  # Prediction
    # outputFromPredict = trainedModel.predict(testFrames, batch_size=(len(testFrames) if len(testFrames) < batch_size else batch_size), steps=testFrames.shape[0]//batch_size,)
    outputFromPredict = np.squeeze(outputFromPredict)  # Remove unnecessary axis
    np.savetxt(predictionsFilename, outputFromPredict, fmt='%f')  

    return outputFromPredict  

def testmodel(test_frames, trained_model: Sequential):
    print("Initial test_frames shape:", test_frames.shape)

    # Address potential deserialization issues properly if needed
    # Investigate why unsafe deserialization might be needed and if 
    # there is a safer alternative 
    
    test_frames = pad_batch(test_frames, batch_size)
    print("Padded test_frames shape:", test_frames.shape)

    try:
        predictions = trained_model.predict(test_frames, batch_size=batch_size)
    except ValueError as e:
        print("Error during prediction:", e)
        # Handle ValueError specifically, possibly adjusting batch size
    except Exception as e:
        raise e  # Raise other exceptions for further inspection

    print("Predictions shape:", predictions.shape)

    predictions = np.squeeze(predictions, axis=1)
    np.savetxt(predictionsFilename, predictions, fmt='%f')

    return predictions

# testmodel will test the model on the test data and create a prediction file
def testmodel_orig(testFrames, trainedModel: Sequential):

    print("testFrames.shape = ", testFrames.shape)
    # config.enable_unsafe_deserialization()    
    
    testFrames = pad_batch(testFrames, batch_size)
    print("testFrames.shape = ", testFrames.shape)
    
    # backend.standardize_dtype(dtype="int16", dtype2="float16")
    # testFrames = np.squeeze(testFrames, axis=1)
    # (len(testFrames) if len(testFrames) < batch_size else batch_size)
    try:
        # outputFromPredict = trainedModel.predict(testFrames, batch_size=batch_size)  # Prediction

        outputFromPredict = trainedModel.predict(testFrames, batch_size=(len(testFrames) if len(testFrames) < batch_size else batch_size), steps=testFrames.shape[0]//batch_size,)
    
    except Exception as e:
        raise e
    
    #  steps=testFrames.shape[0]//batch_size,
    print("outputFromPredict.shape = ", outputFromPredict.shape)
    
    outputFromPredict = np.squeeze(outputFromPredict, axis=1)
    np.savetxt(predictionsFilename, outputFromPredict, fmt='%f')
    
    return outputFromPredict

def checkIfModelExists(modelFile = modelFile):
    """ If the model file exists, return the functional model, ready to predict.  Otherwise return None. """
    print("modelFile = ", modelFile)
    
    # print the current project working directory
    # print("os.getcwd() = ", os.getcwd())
    if os.path.exists(modelFile):
        print("WARNING: Loading model from pre-existing file.  Delete the file to retrain the model from scratch.")
        
        loaded_model: Sequential = None
        try:
            
            # loaded_model.build()
            loaded_model = loadModelFromDisk(modelFile)
            
        except Exception as e:
            raise e
        
        print("loaded_model")
        loaded_model.summary()
        return (True, loaded_model)
    else:
        return (False, None)
    
def speed_test() -> bool:
    model: Sequential = None
    try:
        exists, model = checkIfModelExists(modelFile=modelFile)    
    except Exception as e:
        raise e

    if (model is None):
        print("Model does not exist, creating model...")
        print("Loading training data...")
        frames, train = speedTestCV.loadTrainingData(trainingVideo, trainingSpeeds, batch_size, testmode)

        print("frames.shape = ", frames.shape)
        print("train.shape = ", train.shape)

        print("Loading model...")
        model = createModel(batch_size=batch_size)

        print("Training model...")
        try:
            model = trainModel(model, frames, train)

        except Exception as e:
            raise e
        
        print("Training complete...")
        saveModel(model, modelFile)
        print("Model saved to disk...")
    
    print("Loading Testing data...")
    testFrames = speedTestCV.loadTestingData(testFilename, testmode=testmode)

    print("testFrames.shape before test = ", testFrames.shape)
    
    print("Testing complete, creating predictions...")
    predictions = testmodel(testFrames, model)
    
    # describe the predictions
    print("predictions.shape = ", predictions.shape)
    # print("predictions = ", predictions)

    filename = speedTestCV.publishPredictions(predictions, testFrames, filename="predictions.mp4")
    speedTestCV.displayVideo(filename)
    
    return True



# processNewVideo will process a new video file, editing the video in place
@shared_task
def processNewVideo(filename):
    err = "testing 123"
    
    try:
        modelExists, model = checkIfModelExists(modelFile)
    except Exception as e:
        raise e
    

    if (modelExists is False):
        return "Model does not exist, please train the model first."

    print("Loading data...")
    testFrames = speedTestCV.loadTestingData(filename, True)

    print("Testing complete, creating predictions...")
    predictions = testmodel(testFrames, model)
    
    # describe the predictions
    print("predictions.shape = ", predictions.shape)
    # print("predictions = ", predictions)
    
    _ = speedTestCV.publishPredictions(predictions, testFrames, filename)
    
    return err

def test_single_image():
    frames = speedTestCV.loadTestingData()
    
    # Remove the second dimension
    frames = np.squeeze(frames, axis=1)
    print("frames.shape = ", frames.shape)
    
    image = frames[400]
    superImposedImage = speedTestCV.superimposeImage(18.01, image)
    # display_image("superimposed", superImposedImage)
    imageFinal = speedTestCV.image_with_lanes(superImposedImage)
    # display_image("final", imageFinal)
    return imageFinal
    
# Lane finding Pipeline based on https://www.kaggle.com/code/soumya044/lane-line-detection


if __name__ == "__main__":
    speed_test()
