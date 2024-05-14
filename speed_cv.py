### speedTest - speed_cv.py
### Author: Dave Norton and All the Others/AI Helping (Thank you)! 
# 
# version history:
# 
# 0.001 - feb 2024: initial version
#
# -----
#

import dis
from turtle import right
import numpy as np
import cv2
import math

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
    print("img.shape in draw_lines = ", img.shape)
    
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

# Grayscale the image
def grayscale(img):
    # img = np.squeeze(img, axis=0)
    print("img.shape in grayscale = ", img.shape)
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

    print(speeds_file)
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



# def weighted_average_line(lines):
#     """Calculates a weighted average of lines."""
#     if not lines:
#         return None  

#     weights = np.linspace(1, 0.5, len(lines))  # More weight to recent lines
#     weighted_lines = np.array([line * w for line, w in zip(lines, weights)])
#     return np.sum(weighted_lines, axis=0).astype(np.int32)[0]

def weighted_average_line(lines, prev_lines):
    if not lines:
        return None

    # Calculate the distance between each line and the previous lines
     
    if prev_lines is None:
        prev_lines = np.zeros_like(lines)
    
    distances = np.array([np.min(np.linalg.norm(line - prev_lines, axis=1)) for line in lines])

    # Calculate the weights based on the distances
    weights = np.exp(-distances / np.mean(distances))

    # Normalize the weights
    weights /= np.sum(weights)

    # Calculate the weighted average line
    weighted_lines = np.array([line * w for line, w in zip(lines, weights)])
    return np.sum(weighted_lines, axis=0).astype(np.int32)[0]


def weighted_average_line_a(lines, prev_lines):
    if not lines:
        return None
    
    # Calculate the distance between each line and the previous lines
    distances = np.array([np.min(np.linalg.norm(line - prev_lines, axis=1)) for line in lines])

    # Calculate the weights based on the distances
    weights = np.exp(-distances / np.mean(distances))

    # Normalize the weights
    weights /= np.sum(weights)

    # Calculate the weighted average line
    weighted_lines = np.array([line * w for line, w in zip(lines, weights)])
    return np.sum(weighted_lines, axis=0).astype(np.int32)[0]

def combine_frame_and_overlay(frame, overlay):
    return cv2.addWeighted(frame, 1, overlay, 0.7, 0)

def preprocess_frame(frame):
    shrunken = shrink_img(frame)
    converted = shrunken.astype(np.uint8)
    gray = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Region of interest - return an image of only the region of interest, inside the vertices
def region_of_interest(img, vertices):
    # print("img.shape in region_of_interest = ", img.shape)
    print("vertices = ", vertices)
    # Define a blank mask to start with
    
    # print("img.shape in region_of_interest = ", img.shape)
    mask = np.zeros_like(img)

    # give me the OPPOSITE of the mask, the inverse
    # antimask = cv2.bitwise_not(mask)

    # Fill the mask
    cv2.fillPoly(mask, vertices, 255)

    # Return the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def define_roi(frame):
    print("define_roi: frame.shape = ", frame.shape)
    height = frame.shape[0]
    width = frame.shape[1]

    print("height = ", height)
    print("width = ", width)
    vertices = np.array(
        [[(0, height), (width, height), (width/2, height*0.35)]], dtype=np.int32)
    return vertices

# def define_roi(frame):
#     height = frame.shape[0]
#     width = frame.shape[1]
#     vertices = np.array(
#         [[(0, height-50), (width, height-50), (width/2, height*0.8)]], dtype=np.int32)
#     return vertices

# def define_roi(frame):
#     height = frame.shape[0]
#     width = frame.shape[1]
#     vertices = np.array(
#         [[(0, height*0.6), (width, height*0.6), (width/2, height)]], dtype=np.int32)
    
#     return vertices


def detect_lines(edges, roi):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=70, maxLineGap=20)
    return lines

def filter_lines(lines, slope_threshold):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0:
            slope = abs(dy / dx)
            if slope > slope_threshold:
                filtered_lines.append(line)
    return filtered_lines

def separate_lines(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            # left_lines.append(line)
            print("skip left_lines.append(line) = ", line)
        else:
            right_lines.append(line)
    return left_lines, right_lines

def calculate_weighted_average_line(lines):
    if not lines:
        return None
    weights = np.array([i for i in range(len(lines))])
    weights = weights / weights.sum()
    weighted_lines = np.array([line[0] for line in lines])
    weighted_average_line = np.dot(weights, weighted_lines)
    return weighted_average_line

# def calculate_weighted_average_line(lines):
#     if len(lines) == 0:
#         return None

#     weights = np.linspace(0.1, 1, len(lines))
#     weights = weights / np.sum(weights)

#     weighted_sum_x1 = np.sum([line[0][0] * weight for line, weight in zip(lines, weights)])
#     weighted_sum_y1 = np.sum([line[0][1] * weight for line, weight in zip(lines, weights)])
#     weighted_sum_x2 = np.sum([line[0][2] * weight for line, weight in zip(lines, weights)])
#     weighted_sum_y2 = np.sum([line[0][3] * weight for line, weight in zip(lines, weights)])

#     return [(int(weighted_sum_x1), int(weighted_sum_y1), int(weighted_sum_x2), int(weighted_sum_y2))]


def draw_lines(overlay, lines):
    print("overlay.shape = ", overlay.shape)
    print("lines = ", lines)
    for line in lines:
        print("line = ", line)
        if line is None:
            continue
        if len(line) == 0:
            continue
        x1, y1, x2, y2 = line # for line in lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # x1, y1, x2, y2 = line
        # cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

def add_text(overlay, text):
    cv2.putText(overlay, str(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 255), 2)

def publish_predictions(lane_predictions, video_frames, filename, slope_threshold=0.4):
    num_frames = len(video_frames)
    newVideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (320, 240))
    previous_left_lines = None
    previous_right_lines = None
    for i in range(num_frames):
        frame = video_frames[i]
        edges = preprocess_frame(frame)
        print("edges.shape = ", edges.shape, edges.dtype)

        vertices = define_roi(edges)
        print("vertices = ", vertices)
        roi = region_of_interest(edges, vertices)
        print("roi.shape = ", roi.shape, roi.dtype)

        lines = detect_lines(edges, roi)
        overlay = np.zeros_like(frame)
        
        if lines is not None:
            filtered_lines = filter_lines(lines, slope_threshold)
            left_lines, right_lines = separate_lines(filtered_lines)
            left_line = weighted_average_line(left_lines, previous_left_lines)
            print("left_line = ", left_line)
            right_line = weighted_average_line(right_lines, previous_right_lines)

            previous_left_lines = left_lines
            previous_right_lines = right_lines

            draw_lines(overlay, [left_line, right_line])

        add_text(overlay, lane_predictions[i])
        combined_frame = combine_frame_and_overlay(frame, overlay)
        newVideo.write(combined_frame)

    newVideo.release()
    return filename

# Currently unused
def filter_lines_by_slope(lines, slope_threshold):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        # Calculate slope (avoid division by zero)
        if dx != 0: 
            slope = abs(dy / dx)
            if slope > slope_threshold:
                filtered_lines.append(line)

    return filtered_lines

# currently unused
def get_smoothed_lines(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)

    # Weighted averaging (more recent lines get more weight)
    left_line = weighted_average_line(left_lines)
    right_line = weighted_average_line(right_lines)

    return left_line, right_line

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