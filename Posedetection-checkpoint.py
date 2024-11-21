from time import time
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

#initializing mediapipe class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(1)

results = pose.process(cv2.cvtColor(video,cv2.COLOR_BGR2RGB))
#check if any landmarks is found
if results.pose_landmarks:
    mp_drawing.pose_landmarks(video,landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
    fig = plt.figure(figsize=[10,10])
    #display the output image
    plt.title('output');plt.axis('off');plt.imshow(video[:,:,::-1]);plt.show()
#pose landmarks on 3d
mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)
 
def detectPose(image,pose,display=True):
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    # Initialize the VideoCapture object to read from the webcam.
    video = cv2.VideoCapture(1)
    # Create named window for resizing purposes
    cv2.namedWindow('Pose Detection' ,cv2.WINDOW_NORMAL)
    # Initialize the Videocapture object to read from a video stored in the disk.
    #video = cv2. Videocapture('media/running.mp4')
    # Set video camera size
    video.set(3,1280)
    video.set(4,960)
    # Initialize a variable to store the time of the previous frame.
    time1 = 0
    #iterating through the frames
    while video.isOpened():
        #reading frame by frame
        ok, frame = video.read()
        if not ok:
            break
        #flip the image
        frame = cv2.flip(frame,1)
        frame_height,frame_width, _ = frame.shape
        frame = cv2.resize(frame,(int(frame_width*(640/frame_height))))
        frame, _ = detectPose(frame,pose_video,display = False)
        time2 =time()
        if (time2-time1)>0:
            frames_per_second = 1.0/(time2-time1)
            cv2.putText(frame,'FPS:{}'.format(int(frames_per_second)),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
        time1 = time2
        cv2.imshow('Pose Detection',frame)
        k = cv2.waitKey(1)&0xFF
        if(k==27):
            break
video.release()
cv2.destroyAllWindows()
