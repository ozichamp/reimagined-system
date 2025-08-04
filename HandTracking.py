import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

class HandTrackingKeyDetector:
    """
    Optimized hand tracking system for detecting key presses using MediaPipe and YOLO
    """
    
    def __init__(self, yolo_model_path="D:/google collab rew/best.pt"):
        """
        Initialize the hand tracking and key detection system
        
        Args:
            yolo_model_path (str): Path to the YOLO model file
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand tracking configuration for optimal performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use lighter model for better performance
        )
        
        # YOLO setup
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"YOLO model loaded successfully from: {yolo_model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
        
        # Key detection parameters
        self.distance_threshold = 45  # pixels
        self.z_threshold = -0.05  # depth threshold for key press
        self.movement_threshold = 10  # minimum movement for valid press
        
        # Text display settings
        self.typed_text = []
        self.text_position = (50, 50)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.5
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black
        self.padding = 20
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Previous landmark positions for movement detection
        self.prev_landmarks = {}
        
        # Key detection state
        self.key_press_cooldown = {}  # To prevent multiple detections
        self.cooldown_duration = 0.5  # seconds
        
        print("HandTrackingKeyDetector initialized successfully")
    
    def detect_keys_yolo(self, frame):
        """
        Detect keyboard keys using YOLO model
        
        Args:
            frame: Input video frame
            
        Returns:
            list: Detected key boxes with labels and confidence scores
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO detection with optimized settings
            results = self.yolo_model(
                frame,
                conf=0.3,  # Confidence threshold
                imgsz=320,  # Smaller image size for speed
                verbose=False
            )
            
            detected_keys = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        if hasattr(result, 'names') and class_id in result.names:
                            class_name = result.names[class_id]
                            
                            # Filter out 'keyboard' label, focus on individual keys
                            if class_name.lower() != 'keyboard':
                                detected_keys.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'label': class_name,
                                    'confidence': confidence,
                                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                })
            
            return detected_keys
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            float: Distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_key_press(self, landmark, landmark_id, hand_id, detected_keys, current_time):
        """
        Detect if a landmark is pressing a key
        
        Args:
            landmark: MediaPipe landmark object
            landmark_id: ID of the landmark
            hand_id: ID of the hand (0 or 1)
            detected_keys: List of detected key boxes
            current_time: Current timestamp
            
        Returns:
            str or None: Detected key label if press detected, None otherwise
        """
        # Check cooldown
        cooldown_key = f"{hand_id}_{landmark_id}"
        if cooldown_key in self.key_press_cooldown:
            if current_time - self.key_press_cooldown[cooldown_key] < self.cooldown_duration:
                return None
        
        # Check depth (z-coordinate) for key press
        if landmark.z > self.z_threshold:
            return None
        
        # Convert landmark coordinates to pixel coordinates
        landmark_pixel = (int(landmark.x * 640), int(landmark.y * 480))  # Assuming 640x480 resolution
        
        # Check movement from previous position
        prev_key = f"{hand_id}_{landmark_id}"
        if prev_key in self.prev_landmarks:
            prev_pos = self.prev_landmarks[prev_key]
            movement = self.calculate_distance(landmark_pixel, prev_pos)
            if movement < self.movement_threshold:
                return None
        
        # Update previous position
        self.prev_landmarks[prev_key] = landmark_pixel
        
        # Check distance to detected keys
        for key in detected_keys:
            distance = self.calculate_distance(landmark_pixel, key['center'])
            if distance <= self.distance_threshold:
                # Key press detected!
                self.key_press_cooldown[cooldown_key] = current_time
                print(f"Landmark {hand_id}-{landmark_id} detected: {key['label']}")
                return key['label']
        
        return None
    
    def draw_landmarks(self, frame, hand_landmarks, hand_id):
        """
        Draw hand landmarks and connections on the frame
        
        Args:
            frame: Video frame
            hand_landmarks: MediaPipe hand landmarks
            hand_id: Hand identifier (0 or 1)
        """
        # Draw landmarks and connections
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Highlight fingertips with larger circles
        fingertip_ids = [4, 8, 12, 16, 20]  # THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
        
        for tip_id in fingertip_ids:
            if tip_id < len(hand_landmarks.landmark):
                landmark = hand_landmarks.landmark[tip_id]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green circles for fingertips
    
    def draw_detected_keys(self, frame, detected_keys):
        """
        Draw detected key bounding boxes on the frame
        
        Args:
            frame: Video frame
            detected_keys: List of detected keys
        """
        for key in detected_keys:
            x1, y1, x2, y2 = key['bbox']
            label = key['label']
            confidence = key['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label and confidence
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       self.font, 0.5, (255, 0, 0), 1)
    
    def draw_text_display(self, frame):
        """
        Draw the typed text on the frame with background
        
        Args:
            frame: Video frame
        """
        if not self.typed_text:
            return
        
        # Prepare text
        display_text = ''.join(self.typed_text[-20:])  # Show last 20 characters
        if not display_text:
            return
        
        # Calculate text size
        text_size = cv2.getTextSize(display_text, self.font, self.font_scale, self.font_thickness)[0]
        
        # Calculate background rectangle
        bg_x1 = self.text_position[0] - self.padding
        bg_y1 = self.text_position[1] - text_size[1] - self.padding
        bg_x2 = self.text_position[0] + text_size[0] + self.padding
        bg_y2 = self.text_position[1] + self.padding
        
        # Draw background rectangle
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.bg_color, -1)
        
        # Draw text
        cv2.putText(frame, display_text, self.text_position, 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
    
    def calculate_fps(self):
        """
        Calculate and update FPS
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_fps(self, frame):
        """
        Draw FPS counter on the frame
        
        Args:
            frame: Video frame
        """
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                   self.font, 0.7, (0, 255, 255), 2)
    
    def process_frame(self, frame):
        """
        Process a single frame for hand tracking and key detection
        
        Args:
            frame: Input video frame
            
        Returns:
            frame: Processed frame with annotations
        """
        current_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Detect keys with YOLO
        detected_keys = self.detect_keys_yolo(frame)
        
        # Draw detected keys
        self.draw_detected_keys(frame, detected_keys)
        
        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw landmarks
                self.draw_landmarks(frame, hand_landmarks, hand_id)
                
                # Check each landmark for key presses
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    detected_key = self.detect_key_press(
                        landmark, landmark_id, hand_id, detected_keys, current_time
                    )
                    
                    if detected_key:
                        self.typed_text.append(detected_key)
        
        # Draw text display
        self.draw_text_display(frame)
        
        # Calculate and draw FPS
        self.calculate_fps()
        self.draw_fps(frame)
        
        return frame
    
    def run(self):
        """
        Main execution loop
        """
        print("Starting hand tracking key detection...")
        print("Press 'q' to quit, 'c' to clear typed text")
        
        # Initialize camera
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Error: Camera initialization failed - {e}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Hand Tracking Key Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    self.typed_text.clear()
                    print("Typed text cleared")
        
        except Exception as e:
            print(f"Error: Runtime error - {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Resources released successfully")

def main():
    """
    Main function to run the hand tracking key detector
    """
    try:
        detector = HandTrackingKeyDetector()
        detector.run()
    except Exception as e:
        print(f"Error: Failed to initialize detector - {e}")

if __name__ == "__main__":
    main()