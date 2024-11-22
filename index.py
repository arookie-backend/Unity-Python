from flask import Flask, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

def generate_frames():
    while True:
        success, frame = video_capture.read()  # Read the frame from the camera
        if not success:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the correct format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_game', methods=['POST'])
def control_game():
    command = request.json.get('command')
    # Process the command to control the game
    # For example, you can send commands to Unity via WebSocket or other means
    return jsonify({"status": "success", "command": command})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all interfaces