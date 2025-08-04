#!/usr/bin/env python3
"""
Optimized Hand Tracking with Key Press Detection
================================================
This script uses MediaPipe for hand tracking (21 landmarks per hand) and YOLO for keyboard detection.
It provides real-time key press detection based on finger position and depth analysis.

Requirements:
- cv2 (OpenCV)
- mediapipe
- ultralytics (YOLO)
- numpy
- time

Author: AI Assistant
Date: 2024
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO


class OptimizedHandTracker:
    """
    Optimized hand tracking system with key press detection capabilities.
    """
    
    def __init__(self, model_path="D:/google collab rew/best.pt"):
        """
        Initialize the hand tracker with optimized settings.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Optimized MediaPipe settings for performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use lighter model for better performance
        )
        
        # YOLO model initialization
        try:
            self.yolo_model = YOLO(model_path)
            print(f"YOLO model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
        
        # Key detection parameters
        self.distance_threshold = 45  # Distance threshold for key press detection
        self.z_threshold = -0.02  # Z-coordinate threshold for depth detection
        self.confidence_threshold = 0.3  # YOLO confidence threshold
        self.img_size = 320  # YOLO image size for faster processing
        
        # Text display and tracking
        self.typed_text = []
        self.last_detection_time = {}  # Prevent rapid repeated detections
        self.detection_cooldown = 0.5  # Cooldown period in seconds
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Landmark indices for fingertips
        self.fingertip_landmarks = [4, 8, 12, 16, 20]  # THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
        
        print("Hand tracker initialized successfully!")
        print(f"Distance threshold: {self.distance_threshold} pixels")
        print(f"Z-coordinate threshold: {self.z_threshold}")
        print(f"YOLO confidence threshold: {self.confidence_threshold}")

    def detect_keyboard_keys(self, frame):
        """
        Detect keyboard keys using YOLO model.
        
        Args:
            frame: Input video frame
            
        Returns:
            list: List of detected key bounding boxes and labels
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO detection with optimized settings
            results = self.yolo_model(frame, 
                                    conf=self.confidence_threshold, 
                                    imgsz=self.img_size,
                                    verbose=False)
            
            detected_keys = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                        
                        # Filter out 'keyboard' class, focus on individual keys
                        if class_name.lower() != 'keyboard' and confidence >= self.confidence_threshold:
                            detected_keys.append({
                                'bbox': (x1, y1, x2, y2),
                                'label': class_name,
                                'confidence': confidence,
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                            })
            
            return detected_keys
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            float: Distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_finger_pressing_key(self, landmark, key_info, hand_idx, landmark_idx):
        """
        Check if a finger landmark is pressing a key based on distance and depth.
        
        Args:
            landmark: MediaPipe hand landmark
            key_info: Dictionary containing key information
            hand_idx: Hand index (0 or 1)
            landmark_idx: Landmark index (0-20)
            
        Returns:
            bool: True if key press detected
        """
        # Convert landmark coordinates to pixel coordinates
        landmark_x = int(landmark.x * self.frame_width)
        landmark_y = int(landmark.y * self.frame_height)
        landmark_z = landmark.z
        
        # Calculate distance to key center
        key_center = key_info['center']
        distance = self.calculate_distance((landmark_x, landmark_y), key_center)
        
        # Check distance threshold
        if distance > self.distance_threshold:
            return False
        
        # Check z-coordinate for depth (finger moving forward)
        if landmark_z > self.z_threshold:
            return False
        
        # Check cooldown to prevent rapid repeated detections
        detection_key = f"{hand_idx}_{landmark_idx}_{key_info['label']}"
        current_time = time.time()
        
        if detection_key in self.last_detection_time:
            if current_time - self.last_detection_time[detection_key] < self.detection_cooldown:
                return False
        
        # Update last detection time
        self.last_detection_time[detection_key] = current_time
        
        return True

    def process_key_detection(self, hand_landmarks_list, detected_keys):
        """
        Process key detection for all hands and landmarks.
        
        Args:
            hand_landmarks_list: List of hand landmarks from MediaPipe
            detected_keys: List of detected keys from YOLO
        """
        if not detected_keys:
            return
        
        for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
            for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                # Check all landmarks, but prioritize fingertips
                for key_info in detected_keys:
                    if self.is_finger_pressing_key(landmark, key_info, hand_idx, landmark_idx):
                        # Key press detected!
                        key_label = key_info['label']
                        self.typed_text.append(key_label)
                        
                        # Log the detection
                        print(f"Landmark {hand_idx}-{landmark_idx} detected: {key_label}")
                        
                        # Limit typed text length to prevent memory issues
                        if len(self.typed_text) > 50:
                            self.typed_text = self.typed_text[-50:]

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks and connections on the frame.
        
        Args:
            frame: Video frame
            hand_landmarks: MediaPipe hand landmarks
        """
        # Draw hand landmarks with green color
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Highlight fingertips with larger circles
        for idx in self.fingertip_landmarks:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * self.frame_width)
            y = int(landmark.y * self.frame_height)
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green circles for fingertips

    def draw_detected_keys(self, frame, detected_keys):
        """
        Draw detected key bounding boxes on the frame.
        
        Args:
            frame: Video frame
            detected_keys: List of detected keys
        """
        for key_info in detected_keys:
            x1, y1, x2, y2 = key_info['bbox']
            label = key_info['label']
            confidence = key_info['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label and confidence
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def draw_typed_text(self, frame):
        """
        Draw the typed text on the frame with black background and white text.
        
        Args:
            frame: Video frame
        """
        if not self.typed_text:
            return
        
        # Prepare text
        text = ''.join(self.typed_text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        padding = 20
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Define text position
        text_x = 50
        text_y = 50
        
        # Draw black background rectangle
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_height - padding
        bg_x2 = text_x + text_width + padding
        bg_y2 = text_y + baseline + padding
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Draw white text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    def draw_fps(self, frame):
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Video frame
        """
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def update_fps(self):
        """
        Update FPS counter.
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def run(self):
        """
        Main execution loop for the hand tracking system.
        """
        # Initialize camera
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully!")
            print("Press 'q' to quit, 'c' to clear typed text")
            
        except Exception as e:
            print(f"Error: Camera initialization failed - {e}")
            return
        
        while True:
            try:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                
                # Store frame dimensions
                self.frame_height, self.frame_width = frame.shape[:2]
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand detection
                hand_results = self.hands.process(rgb_frame)
                
                # Detect keyboard keys with YOLO
                detected_keys = self.detect_keyboard_keys(frame)
                
                # Process hand landmarks and key detection
                if hand_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Process key detection
                    self.process_key_detection(hand_results.multi_hand_landmarks, detected_keys)
                
                # Draw detected keys
                self.draw_detected_keys(frame, detected_keys)
                
                # Draw typed text
                self.draw_typed_text(frame)
                
                # Update and draw FPS
                self.update_fps()
                self.draw_fps(frame)
                
                # Display frame
                cv2.imshow('Optimized Hand Tracking - Key Press Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting application...")
                    break
                elif key == ord('c'):
                    self.typed_text.clear()
                    print("Typed text cleared")
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")


def main():
    """
    Main function to run the hand tracking application.
    """
    print("Starting Optimized Hand Tracking System...")
    print("=" * 50)
    
    try:
        # Create and run hand tracker
        tracker = OptimizedHandTracker()
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: Application failed to start - {e}")
    
    print("Hand tracking system terminated")


if __name__ == "__main__":
    main()