import cv2
import mediapipe as mp
import numpy as np

class MovementDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.center_x = None
        self.center_y = None
        self.baseline_set = False
        self.movement_threshold = 0.15 
        
    def set_baseline(self, x, y):
        self.center_x = x
        self.center_y = y
        self.baseline_set = True
        
    def detect_movement(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        movement = 'center'
        h, w, _ = frame.shape
        
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            current_x = (left_shoulder.x + right_shoulder.x) / 2
            current_y = (left_shoulder.y + right_shoulder.y) / 2
            
            if not self.baseline_set:
                self.set_baseline(current_x, current_y)
            
            x_threshold = self.movement_threshold
            y_threshold = self.movement_threshold
            
            x_diff = current_x - self.center_x
            y_diff = current_y - self.center_y
            
            if x_diff < -x_threshold:
                movement = 'left'
            elif x_diff > x_threshold:
                movement = 'right'
            elif y_diff < -y_threshold:
                movement = 'up'
            elif y_diff > y_threshold:
                movement = 'down'
            else:
                movement = 'center'
            
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            if self.baseline_set:
                box_left = int((self.center_x - x_threshold) * w)
                box_right = int((self.center_x + x_threshold) * w)
                box_top = int((self.center_y - y_threshold) * h)
                box_bottom = int((self.center_y + y_threshold) * h)
                
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)
                
                center_pixel = (int(self.center_x * w), int(self.center_y * h))
                cv2.circle(frame, center_pixel, 5, (0, 255, 0), -1)
                
                current_pixel = (int(current_x * w), int(current_y * h))
                cv2.circle(frame, current_pixel, 5, (0, 0, 255), -1)
            
            cv2.putText(frame, f"Movement: {movement.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            movement = None
            
        return movement, frame
    
    def reset_baseline(self):
        self.baseline_set = False
        self.center_x = None
        self.center_y = None


def main():
    cap = cv2.VideoCapture(0)
    detector = MovementDetector()
    
    print("Controls:")
    print("- Stand in center and press 'c' to calibrate")
    print("- Move left/right/up/down to test detection")
    print("- Press 'r' to reset calibration")
    print("- Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        movement, annotated_frame = detector.detect_movement(frame)
        cv2.imshow('Movement Detection', annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.reset_baseline()
            print("Calibrating... Stand in center position")
        elif key == ord('r'):
            detector.reset_baseline()
            print("Baseline reset")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()