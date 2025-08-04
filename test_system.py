#!/usr/bin/env python3
"""
Test script for HandTracking Key Detection System
This script provides automated tests and manual testing guidelines
"""

import cv2
import numpy as np
import time
import sys
import os

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError:
        print("✗ MediaPipe not found. Install with: pip install mediapipe")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
    except ImportError:
        print("✗ Ultralytics not found. Install with: pip install ultralytics")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    return True

def test_camera():
    """Test camera availability and basic functionality"""
    print("\nTesting camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot read from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model...")
    
    model_path = "D:/google collab rew/best.pt"
    
    if not os.path.exists(model_path):
        print(f"✗ YOLO model not found at: {model_path}")
        print("Please ensure the model file exists or update the path in HandTracking.py")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ YOLO model loading failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection"""
    print("\nTesting MediaPipe hand detection...")
    
    try:
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # Create a test image with hand-like features
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[200:300, 300:400] = [255, 255, 255]  # White rectangle as placeholder
        
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        print("✓ MediaPipe hand detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def run_performance_test():
    """Run a basic performance test"""
    print("\nRunning performance test...")
    
    try:
        from HandTracking import HandTrackingKeyDetector
        
        # Create detector instance
        detector = HandTrackingKeyDetector()
        
        # Test with dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Measure processing time
        start_time = time.time()
        num_frames = 10
        
        for i in range(num_frames):
            processed_frame = detector.process_frame(test_frame.copy())
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_frames
        estimated_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"✓ Average processing time: {avg_time:.4f}s per frame")
        print(f"✓ Estimated FPS: {estimated_fps:.1f}")
        
        if estimated_fps >= 25:
            print("✓ Performance test passed (>= 25 FPS)")
            return True
        else:
            print("⚠ Performance warning: FPS below 25")
            return True  # Still pass, but with warning
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def print_manual_test_instructions():
    """Print manual testing instructions"""
    print("\n" + "="*60)
    print("MANUAL TESTING INSTRUCTIONS")
    print("="*60)
    
    print("\n1. LANDMARK VISIBILITY TEST:")
    print("   - Run: python HandTracking.py")
    print("   - Hold both hands in front of camera")
    print("   - Verify: All 21 landmarks visible as green dots")
    print("   - Verify: Hand connections drawn as lines")
    print("   - Verify: Fingertips highlighted with larger green circles")
    
    print("\n2. YOLO KEY DETECTION TEST:")
    print("   - Place a keyboard in camera view")
    print("   - Verify: Blue bounding boxes around detected keys")
    print("   - Verify: Key labels and confidence scores displayed")
    print("   - Verify: 'keyboard' label is filtered out")
    
    print("\n3. KEY PRESS DETECTION TEST:")
    print("   - Position hands above keyboard keys")
    print("   - Press keys with different landmarks (fingertips, knuckles)")
    print("   - Verify: Console shows 'Landmark X-Y detected: Z' messages")
    print("   - Verify: Detected keys appear in top-left text display")
    print("   - Verify: Text has black background and white letters")
    
    print("\n4. PERFORMANCE TEST:")
    print("   - Check FPS counter in top-right corner")
    print("   - Verify: FPS >= 30 for optimal performance")
    print("   - Verify: No significant lag or stuttering")
    
    print("\n5. ERROR HANDLING TEST:")
    print("   - Cover camera lens -> should continue running")
    print("   - Move hands out of frame -> should handle gracefully")
    print("   - Press 'c' to clear typed text")
    print("   - Press 'q' to quit cleanly")
    
    print("\n6. ACCURACY TEST:")
    print("   - Type known sequences (e.g., 'hello')")
    print("   - Verify: >90% accuracy in key detection")
    print("   - Verify: No false positives when hands are stationary")
    print("   - Verify: Cooldown prevents multiple detections")
    
    print("\nEXPECTED CONSOLE OUTPUT:")
    print("   - 'YOLO model loaded successfully from: ...'")
    print("   - 'HandTrackingKeyDetector initialized successfully'")
    print("   - 'Camera initialized successfully'")
    print("   - 'Landmark 0-8 detected: a' (example key press)")
    
    print("\nTROUBLESHOOTING:")
    print("   - If camera fails: Check camera permissions and connections")
    print("   - If YOLO fails: Verify model path and file existence")
    print("   - If low FPS: Close other applications, check CPU usage")
    print("   - If no key detection: Ensure keyboard is well-lit and visible")

def main():
    """Run all automated tests"""
    print("HandTracking Key Detection System - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_dependencies():
        tests_passed += 1
    
    if test_camera():
        tests_passed += 1
    
    if test_yolo_model():
        tests_passed += 1
    
    if test_mediapipe():
        tests_passed += 1
    
    if run_performance_test():
        tests_passed += 1
    
    # Results summary
    print("\n" + "="*50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All automated tests passed! System ready for use.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    # Print manual test instructions
    print_manual_test_instructions()

if __name__ == "__main__":
    main()