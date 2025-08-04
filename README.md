# Hand Tracking Key Detection System

Real-time hand tracking system that detects key presses using MediaPipe and YOLO. The system tracks 21 hand landmarks for both hands and uses computer vision to detect when landmarks interact with keyboard keys.

## Features

- **Real-time Hand Tracking**: Tracks 21 landmarks for both hands using MediaPipe
- **YOLO Key Detection**: Detects individual keyboard keys using custom YOLO model
- **Accurate Key Press Detection**: Uses depth information and distance calculations
- **Performance Optimized**: Runs at 30+ FPS on standard CPU
- **Visual Feedback**: Shows detected keys in real-time with text display
- **Robust Error Handling**: Graceful handling of camera and detection errors

## Requirements

- Python 3.7+
- Webcam/Camera
- YOLO model file (`best.pt`)
- Good lighting conditions
- Visible keyboard in camera frame

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure YOLO model is available:**
   - Place your `best.pt` model file at: `D:/google collab rew/best.pt`
   - Or update the path in `HandTracking.py` line 20

3. **Test system setup:**
   ```bash
   python test_system.py
   ```

## Usage

### Basic Usage

```bash
python HandTracking.py
```

### Controls

- **'q'**: Quit the application
- **'c'**: Clear typed text display
- **ESC**: Alternative quit method

### Expected Behavior

1. **Hand Landmarks**: Green dots and lines showing all 21 hand landmarks
2. **Fingertip Highlights**: Larger green circles on fingertips
3. **Key Detection**: Blue bounding boxes around detected keys
4. **Text Display**: Black box with white text in top-left corner (50, 50)
5. **FPS Counter**: Performance indicator in top-right corner
6. **Console Logging**: Real-time key press detection messages

## System Architecture

### Core Components

1. **HandTrackingKeyDetector Class**: Main system controller
2. **MediaPipe Integration**: 21-point hand landmark detection
3. **YOLO Integration**: Keyboard key detection and classification
4. **Key Press Algorithm**: Distance and depth-based detection
5. **Visual Interface**: Real-time display and feedback

### Key Detection Algorithm

```python
# Detection criteria:
1. Landmark depth (z-coordinate) < threshold (-0.05)
2. Distance to key center < threshold (45 pixels)
3. Sufficient movement from previous position (10+ pixels)
4. Cooldown period to prevent multiple detections (0.5s)
```

### Performance Optimizations

- **Model Complexity**: MediaPipe set to complexity 0 for speed
- **YOLO Settings**: Small image size (320px) and optimized confidence (0.3)
- **Frame Processing**: Efficient landmark-to-pixel coordinate conversion
- **Memory Management**: Proper cleanup and resource management

## Configuration

### Adjustable Parameters

In `HandTracking.py`, you can modify:

```python
# Detection thresholds
self.distance_threshold = 45      # pixels
self.z_threshold = -0.05          # depth threshold
self.movement_threshold = 10      # minimum movement
self.cooldown_duration = 0.5      # seconds

# Display settings
self.text_position = (50, 50)     # text location
self.font_scale = 1.5             # text size
self.padding = 20                 # text background padding
```

### YOLO Model Path

Update the model path in the constructor:

```python
def __init__(self, yolo_model_path="YOUR_MODEL_PATH_HERE"):
```

## Testing

### Automated Tests

Run the test suite to verify system components:

```bash
python test_system.py
```

Tests include:
- Dependency verification
- Camera functionality
- YOLO model loading
- MediaPipe initialization
- Performance benchmarking

### Manual Testing Steps

1. **Landmark Visibility Test**:
   - Hold both hands in front of camera
   - Verify all 21 landmarks are visible
   - Check hand connections and fingertip highlights

2. **Key Detection Test**:
   - Place keyboard in camera view
   - Verify blue bounding boxes around keys
   - Check key labels and confidence scores

3. **Key Press Test**:
   - Position hands above keyboard
   - Press keys with different landmarks
   - Verify console messages and text display

4. **Performance Test**:
   - Monitor FPS counter (should be 30+)
   - Check for lag or stuttering

5. **Accuracy Test**:
   - Type known sequences
   - Verify >90% detection accuracy
   - Test false positive prevention

## Troubleshooting

### Common Issues

**Camera not working:**
- Check camera permissions
- Ensure camera is not used by other applications
- Try different camera indices (0, 1, 2...)

**YOLO model errors:**
- Verify model file exists at specified path
- Check file permissions
- Ensure model is compatible with ultralytics

**Low FPS performance:**
- Close unnecessary applications
- Check CPU usage
- Reduce camera resolution if needed
- Ensure good lighting conditions

**No key detection:**
- Improve lighting conditions
- Position keyboard clearly in frame
- Check YOLO model training data compatibility
- Adjust confidence threshold

**Inaccurate key presses:**
- Calibrate distance threshold
- Adjust depth threshold
- Check hand positioning
- Ensure stable hand movements

### Error Messages

- `"Error: Camera initialization failed"` - Camera access issue
- `"Error loading YOLO model"` - Model file problem
- `"Error in YOLO detection"` - Runtime detection error
- `"Error: Runtime error"` - General processing error

## Performance Expectations

- **FPS**: 30+ on standard CPU
- **Accuracy**: 90%+ key detection accuracy
- **Latency**: <50ms processing per frame
- **Memory**: <500MB RAM usage
- **CPU**: <50% utilization on modern processors

## Technical Details

### Hand Landmarks (MediaPipe)

The system tracks 21 landmarks per hand:
- Wrist (0)
- Thumb: 1-4
- Index finger: 5-8
- Middle finger: 9-12
- Ring finger: 13-16
- Pinky: 17-20

### Coordinate Systems

- **MediaPipe**: Normalized coordinates (0-1)
- **OpenCV**: Pixel coordinates (0-width, 0-height)
- **YOLO**: Bounding box coordinates (x1, y1, x2, y2)

### Detection Pipeline

1. Frame capture from camera
2. MediaPipe hand landmark detection
3. YOLO keyboard key detection
4. Distance calculation between landmarks and keys
5. Depth and movement validation
6. Key press registration and display
7. Visual feedback rendering

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite for diagnostics
3. Review console output for error messages
4. Ensure all dependencies are properly installed