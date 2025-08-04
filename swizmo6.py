import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class TouchlessControl:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Control parameters
        self.active = True
        self.prev_hand_pos = None
        self.prev_hand_x = None
        self.scroll_sensitivity = 27
        self.zoom_sensitivity = 60
        self.zoom_active = False
        self.initial_pinch_dist = None
        
        # Swipe parameters
        self.swipe_threshold = 50
        self.swipe_active = False
        self.swipe_cooldown = 0

        # Screen info
        self.screen_w, self.screen_h = pyautogui.size()

    def track_hand(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Get key points
            wrist = hand_landmarks.landmark[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # For swipe detection
            h, w, c = image.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            hand_x = (wrist_x + index_x) // 2
            hand_y = (wrist_y + index_y) // 2
            
            # Calculate pinch distance (normalized 0-1)
            pinch_dist = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )

            # Check for zoom gesture
            if pinch_dist < 0.15:  # Pinch threshold
                if not self.zoom_active:
                    self.initial_pinch_dist = pinch_dist
                    self.zoom_active = True
                self.handle_zoom(pinch_dist)
            else:
                self.zoom_active = False
                self.handle_movement(wrist)
                self.handle_swipe(hand_x)

            return image, wrist, thumb_tip, index_tip, hand_x, hand_y
        
        self.prev_hand_pos = None
        self.prev_hand_x = None
        self.zoom_active = False
        return image, None, None, None, None, None

    def handle_movement(self, current_pos):
        if not self.active:
            return

        if self.prev_hand_pos is None:
            self.prev_hand_pos = current_pos
            return

        dx = current_pos.x - self.prev_hand_pos.x
        dy = current_pos.y - self.prev_hand_pos.y

        # Horizontal swipe (more sensitive)
        if abs(dx) > 0.012:
            move_x = -dx * self.screen_w * self.scroll_sensitivity
            pyautogui.hscroll(int(move_x/10))

        # Vertical scroll
        if abs(dy) > 0.01:
            move_y = dy * self.screen_h * self.scroll_sensitivity
            pyautogui.scroll(int(-move_y))  # Inverted for natural feel

        self.prev_hand_pos = current_pos

    def handle_zoom(self, current_dist):
        if self.initial_pinch_dist is None:
            return

        zoom_change = (self.initial_pinch_dist - current_dist) * self.zoom_sensitivity

        if abs(zoom_change) > 0.5:
            if zoom_change > 0:  # Fingers moving apart (zoom in)
                pyautogui.keyDown('ctrl')
                pyautogui.press('+')
                pyautogui.keyUp('ctrl')
            else:  # Fingers moving together (zoom out)
                pyautogui.keyDown('ctrl')
                pyautogui.press('-')
                pyautogui.keyUp('ctrl')
            self.initial_pinch_dist = current_dist
            
    def handle_swipe(self, hand_x):
        if self.prev_hand_x is None:
            self.prev_hand_x = hand_x
            return
            
        x_diff = hand_x - self.prev_hand_x
        
        if self.swipe_cooldown <= 0:
            if x_diff < -self.swipe_threshold:
                pyautogui.keyDown('right')  
                pyautogui.keyUp('right')
                self.swipe_active = True
                self.swipe_cooldown = 20  
            elif x_diff > self.swipe_threshold:
                pyautogui.keyDown('left')  
                pyautogui.keyUp('left')
                self.swipe_active = True
                self.swipe_cooldown = 20  
        
        self.prev_hand_x = hand_x
        
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1

def main():
    controller = TouchlessControl()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  
    cap.set(4, 720)   

    print("Touchless Control Active!")
    print("✋ Open hand → Swipe left/right & Scroll up/down")
    print("🤏 Pinch fingers → Zoom in/out")
    print("👆 Horizontal swipe → Arrow key presses")
    print("Press Q to quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame, wrist, thumb, index, hand_x, hand_y = controller.track_hand(frame)

        # Display visual feedback
        if controller.zoom_active:
            color = (0, 255, 255)  # Yellow for zoom
            cv2.putText(frame, "ZOOM MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if thumb and index:
                cv2.line(frame, 
                        (int(thumb.x*frame.shape[1]), int(thumb.y*frame.shape[0])),
                        (int(index.x*frame.shape[1]), int(index.y*frame.shape[0])),
                        color, 2)
        elif controller.swipe_active and controller.swipe_cooldown > 0:
            cv2.putText(frame, "SWIPE DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SWIPE/SCROLL MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if hand_x and hand_y:
            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)

        cv2.putText(frame, "Press Q to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Touchless Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()