import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

#initialssing pose class
mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5, model_complexity=1)
pose_video = mp_pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#creating pose detecting function

def detectPose(image,pose,draw=False,display=False):
    output_image=image.copy()
    imageRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results= pose.process(imageRGB)
    if results.pose_landmarks and draw :
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),thickness=3, circle_radius=3),connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2, circle_radius=2))
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    # Otherwise
    else:
        # Return the output image and the results of pose Landmarks detection.
        return output_image, results

# Read a sample image and perform pose landmarks detection on it.
IMG_PATH = '/Users/Allen/Downloads/istockphoto-174952000-612x612.jpg'
image = cv2.imread(IMG_PATH)
detectPose(image,pose_image,draw=True, display=True)

# camera setup for Pose Detection
'''
# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture (0)
camera_video.set(3,1280)
camera_video.set(4,960)
# Create named window for resizing purposes.
cv2.namedWindow('Pose detection', cv2.WINDOW_NORMAL) 
while camera_video.isOpened() :
        # Read a frame.
        ok, frame = camera_video.read ()
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame of the webcame video.
        frame_height, frame_width, _ = frame.shape
        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)
        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
            cv2.imshow('Pose Detection', frame)
        # Wait for Ims. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0XFF
        # Check if 'ESC is pressed and break the loop.
        if(k == 27):
            break
        # Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllwindows()
'''


def checkHandsJoined(image,results,draw=False,display=False):
    # Get the height and width of the input image.
    height,width, _ = image.shape
    # Create a copy of the input image to write the hands status label on.
    output_image = image.copy()
    # Get the left wrist landmark × and y coordinates.
    left_wrist_landmark =(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    # Get the right wrist landmark x and y coordinates.
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    # Calculate the euclidean distance between the left and right wrist.
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],left_wrist_landmark[1] - right_wrist_landmark[1]))

    # Compare the distance between the wrists with a appropriate threshold to check if both hands are joined.
    if euclidean_distance < 130:
        # Set the hands status to joined.
        hand_status = 'Hands Joined'
        # Set the color value to green.
        color = (0, 255, 0)
    # Otherwise.
    else:
        # Set the hands status to not joined.
        hand_status = 'Hands Not Joined'
        # Set the color value to red.
        color = (0, 0, 255)
    # check if the Hands Joined status and hands distance are specified to be written on the output image.
    if draw:
        # Write the classified hands status on the image.
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        # Write the the distance between the wrists on the image.
        cv2.putText (output_image, f'Distance: (euclidean distance)', (10, 70),cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis ('off');
    # Otherwise
    else:
    # Return the output image and the classified hands status indicating whether the hands are joined or not.
        return output_image, hand_status
    
# Camera setting for CheckHandsjoined
'''
# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture (0)
camera_video.set(3,1280)
camera_video.set(4,960)
# Create named window for resizing purposes.
cv2.namedWindow('Hands Joined?', cv2.WINDOW_NORMAL) 
while camera_video.isOpened() :
        # Read a frame.
        ok, frame = camera_video.read ()
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame of the webcame video.
        frame_height, frame_width, _ = frame.shape
        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)
        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
        # Check if the left and right hands are joined.
            frame, _ = checkHandsJoined(frame, results, draw=True)
        # Display the frame.
        cv2.imshow('Hands Joined?', frame)
        # Wait for Ims. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0XFF
        # Check if 'ESC is pressed and break the loop.
        if(k == 27):
            break
        # Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllwindows()
'''

#Detecting horizontal and vertical movements

def checkleftRight (image, results, draw=False, display=False):
    # Declare a variable to store the horizontal position (left, center, right) of the person.
    horizontal_position = None
    # Get the height and width of the image.
    height, width, _ = image.shape
    # Create a copy of the input image to write the horizontal position on.
    output_image = image.copy()
    # Retreive the x-coordinate of the left shoulder landmark.
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    # Retreive the x-corrdinate of the right shoulder landmark.
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    # Check if the person is at left that is when both shoulder landmarks x-corrdinates
    # are less than or equal to the x-corrdinate of the center of the image.
    if (right_x <= width//2 and left_x <= width//2):
        # Set the person's position to left.
        horizontal_position = 'Left'
    
    # check if the person is at right that is when both shoulder landmarks x-corrdinates
    # are greater than or equal to the x-corrdinate of the center of the image.
    elif (right_x >= width//2 and left_x >= width//2) :
        # Set the person's position to right.
        horizontal_position = 'Right'
    # Check if the person is at center that is when right shoulder landmark x-corrdinate is greater than or equal to
    # # and left shoulder landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
    elif (right_x >= width//2 and left_x <= width//2):
        # Set the person's position to center.
        horizontal_position = 'Center'
    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:
        # Write the horizontal position of the person on the image.
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),3)
        # Draw a Line at the center of the image.
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10,101])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis ('off');

    else:
    #return the output image and horizontal positions
        return output_image, horizontal_position

  
#camera setup for Horizontal Movements 
'''
# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture (0)
camera_video.set(3,1280)
camera_video.set(4,960)
# Create named window for resizing purposes.
cv2.namedWindow('Horizontal Movements', cv2.WINDOW_NORMAL) 
while camera_video.isOpened() :
        # Read a frame.
        ok, frame = camera_video.read ()
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame of the webcame video.
        frame_height, frame_width, _ = frame.shape
        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)
        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
        # Check if the left and right hands are joined.
            frame, _ = checkleftRight(frame, results, draw=True)
        # Display the frame.
        cv2.imshow('Horizontal Movemets?', frame)
        # Wait for Ims. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0XFF
        # Check if 'ESC is pressed and break the loop.
        if(k == 27):
            break
        # Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllwindows()
'''

#vertical Movemants

def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False) :
    # Get the height and width of the image.
    height, width, _ = image.shape
    # Create a copy of the input image to write the posture label on.
    output_image = image.copy ()
    # Retreive the y-coordinate of the left shoulder landmark.
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    # Retreive the y-coordinate of the right shoulder landmark.
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y* height)
    # Calculate the y-coordinate of the mid-point of both shoulders.
    actual_mid_y = abs(right_y + left_y) // 2
    # Calculate the upper and lower bounds of the threshold.
    lower_bound = MID_Y-35
    upper_bound = MID_Y+35
    # Check if the person has jumped that is when the y-coordinate of the mid-point
    # of both shoulders is less than the lower bound.
    if (actual_mid_y < lower_bound):
        # Set the posture to jumping.
        posture = "Jumping"
        # Check if the person has crouched that is when the y-coordinate of the mid-point
        # of both shoulders is greater than the upper bound.
    elif (actual_mid_y > upper_bound):
        # Set the posture to crouching.
        posture = 'Crouching'
    else:
        posture = 'Standing'
    if draw:
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)
    if display:
        plt.figure(figsize= [10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else :
        return output_image,posture
    
#camera setup for Vertical Movements 


# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture (0)
camera_video.set(3,1280)
camera_video.set(4,960)
# Create named window for resizing purposes.
cv2.namedWindow('Vertical Movements', cv2.WINDOW_NORMAL) 
while camera_video.isOpened() :
        # Read a frame.
        ok, frame = camera_video.read ()
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame of the webcame video.
        frame_height, frame_width, _ = frame.shape
        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)
        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
        # Check if the left and right hands are joined.
            frame, _ = checkJumpCrouch(frame, results, draw=True)
        # Display the frame.
        cv2.imshow('Vertical Movemets?', frame)
        # Wait for Ims. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0XFF
        # Check if 'ESC is pressed and break the loop.
        if(k == 27):
            break
        # Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllwindows()


#Pyautogui setup

#UP key
pyautogui.press(keys="up")

#down key
pyautogui.press(keys="down")
