import cv2
import mediapipe as mp
import threading
import time
from flask import Flask, jsonify

app = Flask(__name__)

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.cap = cv2.VideoCapture(0)
        self.current_gesture = "none"
        self.running = False

    def detect_gesture(self):
        success, frame = self.cap.read()
        if not success:
            return "none"

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_TIP]
                index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_MCP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_TIP]
                middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_MCP]

                if (thumb_tip.y < thumb_ip.y and 
                    index_tip.y > index_mcp.y and 
                    middle_tip.y > middle_mcp.y):
                    return "thumbs_up"

        return "none"

    def start_continuous_detection(self):
        self.running = True
        def detection_thread():
            while self.running:
                gesture = self.detect_gesture()
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                time.sleep(0.1)  # Reduce CPU usage

        threading.Thread(target=detection_thread, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.cap.release()

# Initialize the gesture detector
gesture_detector = GestureDetector()

@app.route('/gesture', methods=['GET'])
def get_gesture():
    return jsonify({'gesture': gesture_detector.current_gesture})

@app.route('/start_detection', methods=['GET'])
def start_detection():
    gesture_detector.start_continuous_detection()
    return jsonify({'status': 'Gesture detection started'})

@app.route('/stop_detection', methods=['GET'])
def stop_detection():
    gesture_detector.stop_detection()
    return jsonify({'status': 'Gesture detection stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
