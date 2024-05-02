### speedTest - speed_cv.py
### Author: Dave Norton and All the Others/AI Helping (Thank you)! 
# 
# version history:
# 
# 0.001 - feb 2024: initial version
#
# -----
#
# This file contains the functions for the computer vision part of the speedTest project.
# The functions are used to identify the lanes in the image and draw them on the image.
# The functions are used in the speed_cv.py file.
# The functions are:
# - # display_image: display an image in a window
# - image_with_lanes: identify the lanes in the image and draw them on the image
# - identify_line_slopes: identify the slopes of the contiguous white lines in the image
# - draw_lines: draw the lines on the image
# - region_of_interest: return an image of only the region of interest, inside the vertices
# - grayscale: convert the image to grayscale
# - gaussian_blur: apply Gaussian blur to the image
# - canny: apply Canny edge detection to the image
# - shrink_img: resize the image to 320x240 pixelscheckIfModelExists

import dis
import numpy as np
import cv2
# import keras.backend a
from tensorflow.keras import backend as K

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def image_with_lanes(image):

    # display_image("image", image)

    # Convert the image to grayscale
    gray = grayscale(image)

    print("gray.shape: ", gray.shape, gray.dtype)
    
    # display_image("gray", gray)
    # display_image("image", image)
    
    # Apply Gaussian blur to the image
    blur = gaussian_blur(gray, 5)
    print("blur.shape = ", blur.shape, blur.dtype)

    # display_image("blur", blur)

    # Apply Canny edge detection to the image
    edges = canny(blur, 50, 150)
    print("edges.shape = ", edges.shape, edges.dtype)

    # display_image("edges", edges)

    # Identify the vertices of the wedge shape
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array(
        [[(0, height-10), (width, height-10), (width/2, height*0.45)]], dtype=np.int32)

    # Ignore everything outside the wedge
    roi = region_of_interest(edges, vertices)

    print("roi.shape = ", roi.shape, roi.dtype)
    # display_image("roi", roi)

    roi = np.expand_dims(roi, axis=2)
    print("roi.shape = ", roi.shape, roi.dtype)

    # display_image("roi", roi)

    # Identify contiguous line segments in the image, straight or curved
    lines = identify_line_slopes(roi)
    
    print("before draw_lines image.shape = ", image.shape, image.dtype)

    # Draw the lines on the image
    # display_image("image", image)

    line_image = draw_lines(image, lines)  

    # display_image("line_image", line_image)

    print("line_image.shape = ", line_image.shape, line_image.dtype)
    
    output = line_image
    return output

# Identify the slopes of the contiguous white lines in the image
def identify_line_slopes(img):
    # print ("img.shape in identify_line_slopes = ", img.shape)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # "contours": a matrix with 3 columns:
    # "id": the contour identity (indicates the set of points belonging to the same contour).
    # "x": the x coordinates of the contour points.
    # "y": the y coordinates of the contour points.

    # reduce contours to only contain the two largest contours (most number of matching id's)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # For each line, calculate the average slope
    # empty numpy array to store each line, where each line is a 4-element array
    lines = []
    for contour in contours:
        # print("contour = " + str(contour))
        x, y, w, h = cv2.boundingRect(contour)
        lines.append([x, y, x + w, y + h])      
        # print("lines = " + str(lines))

    return lines

# Draw the lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Create a blank image that matches the original image
    # img = np.squeeze(img, axis=0)
    # print("img.shape in draw_lines = ", img.shape)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    print("line_img.shape = ", line_img.shape)
    print("img.shape in draw_lines = ", img.shape)
    print("lines = ", lines)

    # Does lines contain any data?
    if lines is None:
        return img

    # convert lines to a numpy array
    lines = np.array(lines)

    # Iterate over the lines and draw them on the blank image
    for line in lines:
        line = np.array(line)
        # print("line = " + str(line))
        line_img = cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)

    # print("line_img.shape = ", line_img.shape)
    # print("img.shape in draw_lines = ", img.shape)
    # describe lines, img, and line_img
    print("lines: ", lines.shape, lines.dtype)
    print("img: ", img.shape, img.dtype)
    print("line_img: ", line_img.shape, line_img.dtype)
    
    # Overlay img and line_img
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    # print("img.shape in draw_lines (2) = ", img.shape)  
    
    return img 

# Region of interest - return an image of only the region of interest, inside the vertices
def region_of_interest(img, vertices):
    # Define a blank mask to start with
    
    # print("img.shape in region_of_interest = ", img.shape)
    mask = np.zeros_like(img)

    # Fill the mask
    cv2.fillPoly(mask, vertices, 255)

    # Return the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Grayscale the image
def grayscale(img):
    # img = np.squeeze(img, axis=0)
    # print("img.shape in grayscale = ", img.shape)
    img_bw = img.astype(np.uint8)
    img_bw =  cv2.cvtColor(img_bw, cv2.COLOR_RGB2GRAY)

    return img_bw

# Gaussian blur the image
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def shrink_img(img):
    resized_img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    return resized_img

def load_data(video_file, speeds_file, testmode=False, testmode_num_frames=128, batch_size=32):

    print("Loading data for files: ", video_file, speeds_file)

    # Load the video file
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    if (testmode is True):
        num_frames = testmode_num_frames
        
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("height = ", height)
    print("width = ", width)


    # If the length of the first dimension of frames is less than a multiple of batch_size,
    # change the size of that dimension to a multiple of batch_size
    if (num_frames % batch_size != 0):
        num_frames = (num_frames // batch_size) * batch_size


    print("num_frames = ", num_frames)     

    # Create an array to store the frames using window_nd
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype(np.uint8))

    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading test frame", i)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to grayscale and resize it to a fixed size
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.resize(frame, (320, 240))
        # frames.append(frame)
        frames[i] = shrink_img(frame)

    # frames = np.array(frames)

    # Normalize the pixel values to be between 0 and 1
    # frames = frames / 255.0

    if speeds_file is None:
        speeds = None
    else:
        # Load the speed values
        with open(speeds_file, 'r') as f:
            speeds = [float(line.strip()) for line in f.readlines()]
        speeds = np.array(speeds)
        speeds = speeds[:num_frames]

    frames = np.expand_dims(frames, axis=1)

    cap.release()

    return frames, speeds

# publish_predictions will superimpose the predictions on the testFrames,
# add the lanelines, and display the resulting video 
def publish_predictions_orig(predictions, testFrames, filename):



    newVideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (320, 240))

    # Remove the second dimension
    testFrames = np.squeeze(testFrames, axis=1)
    print("testFrames.shape = ", testFrames.shape, testFrames.dtype)
    print("predictions.shape = ", predictions.shape, predictions.dtype)
    
    for i in range(len(predictions)):
        # superImposedImage = np.squeeze(superImposedImage, axis=1)
        lanelines = image_with_lanes(testFrames[i])
        # display_image("lanelines", lanelines)
        superImposedImage = superimpose_image(predictions[i], lanelines)
        print("superImposedImage.shape = ", superImposedImage.shape, superImposedImage.dtype)
        # display_image("superimposed", superImposedImage)
        # convert the image to uint8 from float
        # imageFinal = imageFinal.astype(np.uint8)
        # print("imageFinal.shape = ", imageFinal.shape, imageFinal.dtype)

        # # display_image("final", imageFinal)
        newVideo.write(superImposedImage)

    # Release the video
    newVideo.release()
    
    return filename

def publish_predictions_b(predictions, test_frames, filename):
    """
    Publishes predictions from a model in video format, processing a video from
    the perspective of a car driver through the windshield.

    Args:
        predictions (np.ndarray): A 1D array of speed predictions (float32).
        test_frames (np.ndarray): A 4D array representing video frames (float32),
                                  with shape (x, 320, 240, 3).

    Returns:
        None (Modifies frames in-place and saves the output video)
    """

    # Check if there are frames and predictions to process
    if len(test_frames) == 0 or len(predictions) == 0:
        print("Error: No frames or predictions provided")
        return

    # Output video setup
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))

    # Remove the second dimension
    test_frames = np.squeeze(test_frames, axis=1)

    # Iterate through frames and predictions
    for i, (frame, prediction) in enumerate(zip(test_frames, predictions)):

        # -- Lane Line Detection --
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur to reduce noise
        canny = cv2.Canny(blur, 50, 150)  # Canny edge detection

        # ... Your lane line fitting logic here ...
        # Assume you have 'left_line' and 'right_line' representing line coordinates

        # -- Superimpose Lines --
        overlay = np.zeros_like(frame)
        cv2.line(overlay, left_line[0], left_line[1], (0, 255, 0), 5)
        cv2.line(overlay, right_line[0], right_line[1], (0, 255, 0), 5)
        frame = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        # -- Add Prediction Text --
        # prediction_text = "Speed: {:.2f}".format(predictions[i])
        # cv2.putText(frame, f"Predicted Speed: {prediction:.2f} km/h", 
        cv2.putText(frame, str(predictions[i]),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # frame looks perfect right now.  Write it to the output video

    out.release()
    return filename

def publish_predictions(predictions, test_frames, filename):

    num_frames = predictions.shape[0] 
    newVideo: cv2.VideoWriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (320, 240))  # Adjust frame rate as needed

    # Remove the second dimension
    test_frames = np.squeeze(test_frames, axis=1)

    for i in range(num_frames):
        frame = test_frames[i]
        overlay = np.zeros_like(frame)

        shrunken = shrink_img(frame)
        converted = shrunken.astype(np.uint8)
        gray = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

        edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds for Canny

        # Region of Interest: Identify the vertices of the wedge shape
        height = edges.shape[0]
        width = edges.shape[1]

        vertices = np.array(
            [[(0, height-100), (width, height-100), (width/2, height*0.35)]], dtype=np.int32)
        
        # Ignore everything outside the wedge
        roi = region_of_interest(edges, vertices)
        
        lines = cv2.HoughLinesP(roi,  # Consider HoughLines for less strict detection
                                1, np.pi / 180, 20, minLineLength=70, maxLineGap=20)

        if lines is not None:
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)  # Calculate the slope
                if slope < 0:  # If the slope is negative, the line is on the left
                    left_lines.append(line)
                else:  # Otherwise, the line is on the right
                    right_lines.append(line)

            left_line = [[]]
            right_line = [[]]
            if len(left_lines) > 0:
                left_line = np.mean(left_lines, axis=0, dtype=np.int32)
            if len(right_lines) > 0:
                right_line = np.mean(right_lines, axis=0, dtype=np.int32)
    
            for line in [left_line, right_line]: 
                if len(line[0]) == 0:
                    continue
                x1, y1, x2, y2 = line[0] # for line in lines:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(overlay, str(predictions[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (100, 0, 255), 2)

        frame = cv2.addWeighted(frame, 1, overlay, 0.7, 0)
        newVideo.write(frame)
        
    newVideo.release()
    return filename

# Open the video and play it
def display_video(filename):
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

def superimpose_image(prediction, image):
    superimposedImage = cv2.putText(image, str(prediction), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # print("superimposedImage.shape = ", superimposedImage.shape)
    return superimposedImage