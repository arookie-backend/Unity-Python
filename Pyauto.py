import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
 
# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
 
# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
 
# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, draw=False, display=False):
    '''
    This function performs the pose detection on the most prominent person in an image.
    Args:
        image:   The input image with a prominent person whose pose landmarks needs to be detected.
        pose:    The pose function required to perform the pose detection.
        draw:    A boolean value that is if set to true the function draw pose landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the 
                 resultant image and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn if it was specified.
        results:      The output of the pose landmarks detection on the input image.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Check if any landmarks are detected and are specified to be drawn.
    if results.pose_landmarks and draw:
    
        # Draw Pose Landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))
 
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    # Otherwise
    else:
 
        # Return the output image and the results of pose landmarks detection.
        return output_image, results

def checkHandsJoined(image, results, draw=False, display=False):
    '''
    This function checks whether the hands of the person are joined or not in an image.
    Args:
        image:   The input image with a prominent person whose hands status (joined or not) needs to be classified.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the hands status &amp; distance on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image: The same input image but with the classified hands status written, if it was specified.
        hand_status:  The classified status of the hands whether they are joined or not.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the hands status label on.
    output_image = image.copy()
    
    # Get the left wrist landmark x and y coordinates.
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    # Get the right wrist landmark x and y coordinates.
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    # Calculate the euclidean distance between the left and right wrist.
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
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
        
    # Check if the Hands Joined status and hands distance are specified to be written on the output image.
    if draw:

        # Write the classified hands status on the image. 
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        # Write the the distance between the wrists on the image. 
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the classified hands status indicating whether the hands are joined or not.
        return output_image, hand_status

def checkHandsJoined(image, results, draw=False, display=False):
    '''
    This function checks whether the hands of the person are joined or not in an image.
    Args:
        image:   The input image with a prominent person whose hands status (joined or not) needs to be classified.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the hands status &amp; distance on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image: The same input image but with the classified hands status written, if it was specified.
        hand_status:  The classified status of the hands whether they are joined or not.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the hands status label on.
    output_image = image.copy()
    
    # Get the left wrist landmark x and y coordinates.
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
 
    # Get the right wrist landmark x and y coordinates.
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    # Calculate the euclidean distance between the left and right wrist.
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
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
        
    # Check if the Hands Joined status and hands distance are specified to be written on the output image.
    if draw:
 
        # Write the classified hands status on the image. 
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        # Write the the distance between the wrists on the image. 
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    # Check if the output image is specified to be displayed.
    if display:
 
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the classified hands status indicating whether the hands are joined or not.
        return output_image, hand_status

def checkLeftRight(image, results, draw=False, display=False):
    '''
    This function finds the horizontal position (left, center, right) of the person in an image.
    Args:
        image:   The input image with a prominent person whose the horizontal position needs to be found.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the horizontal position on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:         The same input image but with the horizontal position written, if it was specified.
        horizontal_position:  The horizontal position (left, center, right) of the person in the input image.
    '''
    
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
    if (right_x &lt;= width//2 and left_x &lt;= width//2):
        
        # Set the person's position to left.
        horizontal_position = 'Left'

    # Check if the person is at right that is when both shoulder landmarks x-corrdinates
    # are greater than or equal to the x-corrdinate of the center of the image.
    elif(right_x &gt;= width//2 and left_x &gt;= width//2):
        # Set the person's position to right.
        horizontal_position = 'Right'
    
    # Check if the person is at center that is when right shoulder landmark x-corrdinate is greater than or equal to
    # and left shoulder landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
    elif (right_x &gt;= width//2 and left_x &lt;= width//2):
        
        # Set the person's position to center.
        horizontal_position = 'Center'
        
    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:

        # Write the horizontal position of the person on the image. 
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Draw a line at the center of the image.
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the person's horizontal position.
        return output_image, horizontal_position

def checkLeftRight(image, results, draw=False, display=False):
    '''
    This function finds the horizontal position (left, center, right) of the person in an image.
    Args:
        image:   The input image with a prominent person whose the horizontal position needs to be found.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the horizontal position on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:         The same input image but with the horizontal position written, if it was specified.
        horizontal_position:  The horizontal position (left, center, right) of the person in the input image.
    '''
    
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
    if (right_x &lt;= width//2 and left_x &lt;= width//2):
        
        # Set the person's position to left.
        horizontal_position = 'Left'
 
    # Check if the person is at right that is when both shoulder landmarks x-corrdinates
    # are greater than or equal to the x-corrdinate of the center of the image.
    elif (right_x &gt;= width//2 and left_x &gt;= width//2):
        
        # Set the person's position to right.
        horizontal_position = 'Right'
    
    # Check if the person is at center that is when right shoulder landmark x-corrdinate is greater than or equal to
    # and left shoulder landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
    elif (right_x &gt;= width//2 and left_x &lt;= width//2):
        
        # Set the person's position to center.
        horizontal_position = 'Center'
        
    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:
 
        # Write the horizontal position of the person on the image. 
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Draw a line at the center of the image.
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
    # Check if the output image is specified to be displayed.
    if display:
 
        # Display the output image.
        plt.figure(figsize=import cv2
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

'''
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

'''
#-------------------------------------------------------------------------------------------------------
#integrating pose with pyautogui
#-------------------------------------------------------------------------------------------------------


camera_video = cv2.VideoCapture (0)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('App Integration', cv2.WINDOW_NORMAL)
time1 = 0
game_started = False 
x_pos_index = 1
y_pos_index = 1
MID_Y=None
counter = 0
num_of_frames = 10
while camera_video.isOpened():
    ok, frame = camera_video.read ()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    # Get the height and width of the frame of the webcame video.
    frame_height, frame_width, _ = frame.shape
    # Perform the pose detection on the frame.
    frame, results = detectPose(frame, pose_video, draw=game_started)
    if results.pose_landmarks:
        if game_started:
            frame, horizontal_position = checkleftRight(frame, results, draw=True)

        if (horizontal_position == 'Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
            pyautogui.press('left')
            x_pos_index-=1
        elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
            pyautogui.press('right')
            x_pos_index+=1
        '''if checkHandsJoined(frame, results) [1] == 'Hands Joined':
            pyautogui.press('space')'''

    else:

        # Command to start the game first time.
        cv2.putText(frame, "JOIN BOTH HANDS TO START THE GAME.", (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0),3)
    # Check if the left and right hands are joined.
    if checkHandsJoined(frame,results)[1] == 'Hands Joined':
        # Increment the count of consecutive frames with +ve condition.
        counter += 1
        if counter == num_of_frames:
            if not(game_started):
                game_started = True
                left_y  = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                MID_Y = abs(right_y + left_y) // 2
                pyautogui.click(x=1300, y=800, button='left')
            else:
                pyautogui.press('space')
            counter = 0
        else:
            counter = 0
    #commands to control the vertical movements
    if MID_Y:
        # Get posture (jumping, crouching or standing) of the person in the frame.
        frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
        # Check if the person has jumped.
        if posture == 'Jumping' and y_pos_index == 1:
        # Press the up arrow key
            pyautogui.press('up')
        # Update the veritcal position index of the character.
            y_pos_index += 1
        # Check if the person has crouched.
        elif posture == 'Crouching' and y_pos_index == 1:
        # Press the down arrow key
            pyautogui.press ('down')
        # Update the veritcal position index of the character.
            y_pos_index -= 1
        # Check if the person has stood.
        elif posture == 'Standing' and y_pos_index != 1:
            # Update the veritcal position index of the character.
            y_pos_index = 1
    else:
        counter=0

    time2 = time()
    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) >0:
    # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        # Write the calculated number of frames per second on the frame.
        cv2.putText(frame, 'FPS: (}'.format(int(frames_per_second)),(10, 30),cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0),3)
        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
    time1 = time2
        # Display the frame.
    cv2.imshow ('Subway Surfers with Pose Detection', frame)
    # Wait for ims. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    # Check if 'ESC is pressed and break the loop.
    if(k == 27):
        break
camera_video.release()
cv2.destroyAllwindows()
 [10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the person's horizontal position.
        return output_image, horizontal_position