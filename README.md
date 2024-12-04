# YOLOv10 Object Detection Demo

This repository contains a web-based demonstration of YOLOv10 object detection capabilities, featuring real-time detection through webcam, image upload, and video processing. The system is deployed on a Google Cloud Compute Engine VM instance equipped with an NVIDIA A100 GPU for high-performance inference.

## System Architecture

- **Backend**: Flask-based Python server handling model inference and image processing
- **Frontend**: HTML/JavaScript web interface for real-time interaction
- **Model**: YOLOv10 implementation with multiple model size variants
- **Infrastructure**: Google Cloud Compute Engine VM instance with NVIDIA A100 GPU
- **Processing**: Real-time object detection for images, videos, and webcam feeds

## Features

### 1. Multiple Model Options
Choose from various YOLOv10 model sizes:
- `yolov10n`: Nano version (fastest, lowest resource usage)
- `yolov10s`: Small version
- `yolov10m`: Medium version (balanced performance)
- `yolov10l`: Large version
- `yolov10x`: Extra large version (highest accuracy)
- `yolov10thermal`: Specialized version for thermal imaging

### 2. Input Methods
The system supports multiple ways to perform object detection:

#### Image Upload
- Upload local image files (supported formats: .jpg, .jpeg, .png)
- Real-time processing and display of detection results

#### Video Upload
- Upload video files (supported formats: .mp4, .avi, .mov)
- Frame-by-frame processing with real-time display
- Adjustable confidence threshold and image size

#### Webcam Detection
- Real-time object detection through browser webcam access
- Continuous frame processing and result display
- Adjustable parameters during active detection

#### Sample Images
- Built-in thermal dataset examples
- Ground truth comparison available for thermal images
- Performance metric calculation for evaluation

### 3. Adjustable Parameters

- **Image Size**: Range from 320 to 1280 pixels (default: 640)
- **Confidence Threshold**: Range from 0 to 1 (default: 0.25)
- **Ground Truth Display**: Toggle for comparing with annotated data

## Performance Metrics

The system provides several metrics for evaluating detection performance:

### IoU (Intersection over Union)
- Measures the overlap between predicted and ground truth bounding boxes
- Range: 0 to 1 (higher is better)
- Calculated as: (Area of Intersection) / (Area of Union)
- Useful for evaluating the accuracy of object localization

### False Positives
- Number of incorrect detections (objects detected where none exist)
- Indicates how often the model makes spurious detections
- Lower values indicate better precision

### False Negatives
- Number of missed detections (existing objects not detected)
- Indicates how often the model fails to detect actual objects
- Lower values indicate better recall

## Setup and Usage

Can be accessed at yolov10demo.live or locally:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov10-demo.git
cd yolov10-demo
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface at `http://localhost:8080`

## Using the Interface

1. **Select Model**: Choose the desired YOLOv10 model variant from the dropdown menu

2. **Adjust Parameters**:
   - Use the Image Size slider to balance between speed and accuracy
   - Adjust the Confidence Threshold to filter detection results

3. **Choose Input Method**:
   - Click "Start Webcam" for real-time webcam detection
   - Use the file input to upload images or videos
   - Browse example images from the thermal dataset

4. **View Results**:
   - Left panel shows input feed
   - Right panel displays detection results
   - Metrics panel shows performance statistics (when applicable)

## Performance Considerations

- Larger model variants (l, x) provide higher accuracy but require more processing time
- Smaller variants (n, s) offer faster inference with reduced accuracy
- Image size affects both detection accuracy and processing speed
- Higher confidence thresholds reduce false positives but may increase false negatives

## Limitations

- Webcam functionality requires browser permission for camera access
- Video processing speed depends on server GPU availability and network conditions
- Ground truth comparison only available for thermal dataset examples


## Acknowledgments

Based on the YOLOv10 implementation (https://github.com/THU-MIG/yolov10)

## Contact

syre@duck.com