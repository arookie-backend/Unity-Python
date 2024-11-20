import flet as ft
import cv2
import mediapipe as mp
import threading
import time

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

    def start_continuous_detection(self, update_callback):
        self.running = True
        def detection_thread():
            while self.running:
                gesture = self.detect_gesture()
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                    update_callback(gesture)
                time.sleep(0.1)  # Reduce CPU usage

        threading.Thread(target=detection_thread, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.cap.release()

def main(page: ft.Page):
    page.title = "Gesture Recognition"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    gesture_detector = GestureDetector()
    
    # UI Components
    gesture_text = ft.Text(value="Current Gesture: None", size=20)
    start_button = ft.ElevatedButton(text="Start Detection")
    stop_button = ft.ElevatedButton(text="Stop Detection", visible=False)

    def update_gesture(gesture):
        gesture_text.value = f"Current Gesture: {gesture}"
        page.update()

    def start_detection(e):
        gesture_detector.start_continuous_detection(update_gesture)
        start_button.visible = False
        stop_button.visible = True
        page.update()

    def stop_detection(e):
        gesture_detector.stop_detection()
        start_button.visible = True
        stop_button.visible = False
        gesture_text.value = "Current Gesture: None"
        page.update()

    start_button.on_click = start_detection
    stop_button.on_click = stop_detection

    # Add controls to the page
    page.add(
        gesture_text,
        ft.Row([start_button, stop_button], alignment=ft.MainAxisAlignment.CENTER)
    )

    # Cleanup on window close
    page.on_close = lambda _: gesture_detector.stop_detection()

# Run the Flet app
ft.app(target=main)