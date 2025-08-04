#!/usr/bin/env python3
"""
System Test Script for Hand Tracking Project
============================================
This script tests all dependencies and system components before running the main application.

Usage: python test_system.py
"""

import sys
import importlib
import cv2
import time

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe', 
        'ultralytics': 'ultralytics',
        'numpy': 'numpy'
    }
    
    failed_imports = []
    
    for package, pip_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package} ({pip_name}) - OK")
        except ImportError as e:
            print(f"✗ {package} ({pip_name}) - FAILED: {e}")
            failed_imports.append(pip_name)
    
    if failed_imports:
        print(f"\nMissing packages. Install with:")
        print(f"pip install {' '.join(failed_imports)}")
        return False
    
    print("All packages imported successfully!\n")
    return True

def test_camera():
    """Test camera accessibility."""
    print("Testing camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera - Cannot open camera (index 0)")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Camera - Cannot read frame from camera")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✓ Camera - OK (Resolution: {width}x{height})")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera - Error: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection."""
    print("Testing MediaPipe hand detection...")
    
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
        
        print("✓ MediaPipe - Hand detection model initialized successfully")
        hands.close()
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe - Error: {e}")
        return False

def test_yolo_availability():
    """Test YOLO model availability (without loading the actual model)."""
    print("Testing YOLO availability...")
    
    try:
        from ultralytics import YOLO
        
        # Test if YOLO class can be instantiated (without loading actual model)
        print("✓ YOLO - ultralytics package available")
        print("Note: Actual model loading will be tested when running HandTracking.py")
        return True
        
    except Exception as e:
        print(f"✗ YOLO - Error: {e}")
        return False

def test_opencv_features():
    """Test OpenCV features used in the project."""
    print("Testing OpenCV features...")
    
    try:
        # Test basic OpenCV functions
        import numpy as np
        
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test drawing functions
        cv2.rectangle(test_img, (10, 10), (50, 50), (255, 0, 0), 2)
        cv2.putText(test_img, "Test", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(test_img, (75, 75), 10, (0, 255, 0), -1)
        
        # Test color conversion
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        print("✓ OpenCV - Drawing and color conversion functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ OpenCV - Error: {e}")
        return False

def performance_benchmark():
    """Run a simple performance benchmark."""
    print("Running performance benchmark...")
    
    try:
        import numpy as np
        
        # Simple array operations benchmark
        start_time = time.time()
        
        for _ in range(1000):
            arr = np.random.rand(100, 100)
            result = np.sqrt(arr * arr + arr * arr)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ Performance - Array operations: {duration:.3f}s (1000 iterations)")
        
        if duration > 2.0:
            print("⚠ Warning: Performance seems slow. Consider closing other applications.")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance - Error: {e}")
        return False

def main():
    """Run all system tests."""
    print("Hand Tracking System Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe", test_mediapipe), 
        ("YOLO Availability", test_yolo_availability),
        ("OpenCV Features", test_opencv_features),
        ("Performance", performance_benchmark)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed_tests += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready to run HandTracking.py")
        print("\nTo start the application, run:")
        print("python HandTracking.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running the main application.")
        
        print("\nTroubleshooting tips:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check camera permissions and connections")
        print("3. Ensure Python version is 3.8+")
        print("4. Close other applications using the camera")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)