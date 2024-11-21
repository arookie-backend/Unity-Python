from flask import Flask, Response,request
from flask import jsonify
import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Initialize pose class
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

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
# IMG_PATH = '/Users/Allen/Downloads/istockphoto-174952000-612x612.jpg'
# image = cv2.imread(IMG_PATH)


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
    lower_bound = MID_Y-70
    upper_bound = MID_Y+55
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

# Function to generate frames with pose detection and game control
generate_frames_flag = False
start_time = None
def generate_frames():
    global generate_frames_flag, start_time
    start_time = time()
    camera_video = cv2.VideoCapture(0)
    #camera_video.set(3,1280)
    #camera_video.set(4,960)

    # Create named window for resizing purposes.
    #cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
    #store the time of the previous frame.
    time1 = 0

    game_started = False   

    # At Start the character is at center so the index is 1 and it can move left (value 0) and right (value 2).
    x_pos_index = 1

    # At Start the person is standing so the index is 1 and he can crouch (value 0) and jump (value 2).
    y_pos_index = 1

    # Declate a variable to store the intial y-coordinate of the mid-point of both shoulders of the person.
    MID_Y = None

    # Initialize a counter to store count of the number of consecutive frames with person's hands joined.
    counter = 0

    # Initialize the number of consecutive frames on which we want to check if person hands joined before starting the game.
    num_of_frames = 10


    while generate_frames_flag:
        
        
        ok, frame = camera_video.read()
        
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape
        
        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=game_started)
        
        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
            
            # Check if the game has started
            if game_started:
                
                # Get horizontal position of the person in the frame.
                frame, horizontal_position = checkleftRight(frame, results, draw=True)
                
                # Check if the person has moved to left from center or to center from right.
                if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                    
                    pyautogui.press('left')
                    x_pos_index -= 1               

                # Check if the person has moved to Right from center or to center from left.
                elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):

                    pyautogui.press('right')
                    x_pos_index += 1  
            else:
                
                # Write the text representing the way to start the game on the frame. 
                cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 3)
            
            # Command to Start or resume the game.
            #------------------------------------------------------------------------------------------------------------------
            
            # Check if the left and right hands are joined.
            if checkHandsJoined(frame, results)[1] == 'Hands Joined':

                # Increment the count of consecutive frames with +ve condition.
                counter += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                if counter == num_of_frames:

                    # Command to Start the game first time.
                    #----------------------------------------------------------------------------------------------------------
                    
                    # Check if the game has not started yet.
                    if not(game_started):

                        # Update the value of the variable that stores the game state.
                        game_started = True

                        # Retreive the y-coordinate of the left shoulder landmark.
                        left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

                        # Retreive the y-coordinate of the right shoulder landmark.
                        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)

                        # Calculate the intial y-coordinate of the mid-point of both shoulders of the person.
                        MID_Y = abs(right_y + left_y) // 2

                        # Move to 1300, 800, then click the left mouse button to start the game.
                        pyautogui.click(x=1300, y=800, button='left')
                    
                
                    # Command to resume the game after death of the character.
                    #----------------------------------------------------------------------------------------------------------
                    
                    # Otherwise if the game has started.
                    else:

                        # Press the space key.
                        pyautogui.press('space')
                    
                    #----------------------------------------------------------------------------------------------------------
                    
                    # Update the counter value to zero.
                    counter = 0

            # Otherwise if the left and right hands are not joined.        
            else:

                # Update the counter value to zero.
                counter = 0
                
            #------------------------------------------------------------------------------------------------------------------

            # Commands to control the vertical movements of the character.
            #------------------------------------------------------------------------------------------------------------------
            
            # Check if the intial y-coordinate of the mid-point of both shoulders of the person has a value.
            if MID_Y:
                
                # Get posture (jumping, crouching or standing) of the person in the frame. 
                frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
                
                # Check if the person has jumped.
                if posture == 'Jumping' and y_pos_index == 1:

                    # Press the up arrow key
                    pyautogui.press('up')
                    
                    # Update the veritcal position index of  the character.
                    y_pos_index += 1 

                # Check if the person has crouched.
                elif posture == 'Crouching' and y_pos_index == 1:

                    # Press the down arrow key
                    pyautogui.press('down')
                    
                    # Update the veritcal position index of the character.
                    y_pos_index -= 1
                
                # Check if the person has stood.
                elif posture == 'Standing' and y_pos_index   != 1:
                    
                    # Update the veritcal position index of the character.
                    y_pos_index = 1
            
            #------------------------------------------------------------------------------------------------------------------
        
        
        # Otherwise if the pose landmarks in the frame are not detected.       
        else:

            # Update the counter value to zero.
            counter = 0
            
        # Calculate the frames updates in one second
        #----------------------------------------------------------------------------------------------------------------------
        time2 = time()
        
        if (time2 - time1) > 0:
        
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        time1 = time2
        
        #----------------------------------------------------------------------------------------------------------------------
        
        # Display the frame.            
        # cv2.imshow('Navigation Control System', frame)
        
        # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) &  0xFF    
        
        # Check if 'ESC' is pressed and break the loop.
        if(k == 27):
            break

    # Release the VideoCapture object and close the windows.
    camera_video.release()
    elapsed_time = time() - start_time
    return f'Elapsed time: {elapsed_time} seconds'
    cv2.destroyAllWindows()

    
@app.route('/stop')
def stop():
    global generate_frames_flag, start_time
    generate_frames_flag = False
    if start_time:
        elapsed_time = (time() - start_time) / 60  # Convert seconds to minutes
        start_time = None
        calories_burned = (elapsed_time * 8 * 65) / 200  # Assuming 8 calories burned per minute for walking, 65kg person
        return f'Calories Burned: {calories_burned}\nTime Played: {elapsed_time} minutes'
    else:
        return 'Frames generation stopped!'
    
@app.route('/calories',methods=['GET'])
def returnCalorie():
    # global generate_frames_flag, start_time
    # generate_frames_flag = False
    d={}
    elapsed_time = (time() - start_time) / 60 
    calories_burned = (elapsed_time * 8 * 65) / 200
    calories_burned_str = str(calories_burned)
    d['output']= calories_burned_str
    return d

@app.route('/api',methods=['GET'])
def returnAscii():
    d={}
    # inputchr = str(request.args.get('query'))
    # answer = str(ord(inputchr))
    elapsed_time = (time() - start_time) / 60 
    calories_burned = (elapsed_time * 8 * 65) / 200
    calories_burned_str = str(calories_burned)
    d['output']= calories_burned_str
    return d
    
        
@app.route('/video_feed')
def video_feed():
    global generate_frames_flag
    generate_frames_flag = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=5001)
