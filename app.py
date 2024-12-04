from flask import Flask, render_template, Response, request, jsonify, session
import cv2
from ultralytics import YOLOv10
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
from queue import Queue
from threading import Thread
import uuid
import torch
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = os.urandom(24)
models = {}
frame_queues = {}
webcam_streams = {}

MODEL_CHOICES = [
    "yolov10n", "yolov10s", "yolov10m",
    "yolov10b", "yolov10l", "yolov10x",
    "yolov10thermal"
]

def get_model(model_id):
    if model_id not in models:
        model_path = os.path.join('models', f'{model_id}.pt')
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found")
        models[model_id] = YOLOv10(model_path)
        if torch.cuda.is_available():
            models[model_id].to('cuda')
    return models[model_id]

def process_frame(frame, model, image_size, conf, do_detection=True):
    # Always resize the frame first
    h, w = frame.shape[:2]
    ratio = min(image_size/w, image_size/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(frame, (new_w, new_h))
    results = model(resized, imgsz=image_size, conf=conf)
    processed_frame = results[0].plot()
    return processed_frame

'''
def webcam_processor(model, image_size, conf, session_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
        
    queue = frame_queues[session_id] = Queue(maxsize=10)
    webcam_streams[session_id] = cap
    frame_count = 0
    
    while session_id in webcam_streams:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        do_detection = True
        processed_frame = process_frame(frame, model, image_size, conf, do_detection)
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, raw_buffer = cv2.imencode('.jpg', frame, encode_params)
        _, processed_buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        
        frame_data = {
            'raw': base64.b64encode(raw_buffer).decode(),
            'processed': base64.b64encode(processed_buffer).decode()
        }
        
        if session_id in webcam_streams:
            try:
                queue.put_nowait(frame_data)
            except:
                _ = queue.get()
                queue.put_nowait(frame_data)
    
    cap.release()

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    try:
        model_id = request.form.get('model', 'yolov10m')
        image_size = int(request.form.get('image_size', 640))
        conf = float(request.form.get('conf', 0.25))
        
        model = get_model(model_id)
        session_id = str(uuid.uuid4())
        
        Thread(target=webcam_processor,
               args=(model, image_size, conf, session_id),
               daemon=True).start()
               
        return jsonify({'session_id': session_id})
    except Exception as e:
        print(f"Webcam start error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    try:
        model = get_model(model_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
        
    session_id = str(uuid.uuid4())
    
    Thread(target=webcam_processor,
           args=(model, image_size, conf, session_id),
           daemon=True).start()
           
    return jsonify({'session_id': session_id})

@app.route('/stop_webcam/<session_id>')
def stop_webcam(session_id):
    if session_id in webcam_streams:
        del webcam_streams[session_id]
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Invalid session'}), 404
'''
    
@app.route('/get_frame/<session_id>')
def get_frame(session_id):
    if session_id not in frame_queues:
        return jsonify({'error': 'Invalid session'}), 404
        
    queue = frame_queues[session_id]
    frame_data = queue.get()
    
    if frame_data is None:
        del frame_queues[session_id]
        return jsonify({'end': True})
    
    # Handle both webcam (dict with raw/processed) and video (single frame) data
    if isinstance(frame_data, dict):
        return jsonify(frame_data)
    else:
        return jsonify({'frame': frame_data})


def video_processor(input_path, model, image_size, conf, session_id):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    queue = frame_queues[session_id] = Queue(maxsize=10)
    
    while cap.isOpened() and session_id in frame_queues:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame, model, image_size, conf)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        frame_data = base64.b64encode(buffer).decode()
        
        try:
            queue.put_nowait(frame_data)
        except:
            _ = queue.get()
            queue.put_nowait(frame_data)
    
    cap.release()
    queue.put(None)  # Signal end of processing
    os.unlink(input_path)

@app.route('/')
def index():
    return render_template('index.html', models=MODEL_CHOICES)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    model_id = request.form.get('model', 'yolov10m')
    image_size = int(request.form.get('image_size', 640))
    conf = float(request.form.get('conf', 0.25))
    show_gt = request.form.get('show_gt', 'true') == 'true'
    is_webcam_frame = request.form.get('is_webcam_frame', 'false') == 'true'

    try:
        model = get_model(model_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    if is_webcam_frame:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img, imgsz=image_size, conf=conf)
        processed = results[0].plot()
        _, buffer = cv2.imencode('.jpg', processed)
        img_str = base64.b64encode(buffer).decode()
        return jsonify({'type': 'image', 'data': img_str})

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_height, image_width = img.shape[:2]

        results = model(img, imgsz=image_size, conf=conf)
        processed = results[0].plot()
        boxes = results[0].boxes

        avg_iou = None

        if 'FLIR_' in file.filename and show_gt:
            label_path = os.path.join('static', 'thermal_dataset', 'labels',
                                      os.path.splitext(file.filename)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    gt_boxes = read_yolo_label(label_path, image_width, image_height)

                # Handle predicted boxes
                pred_boxes = []
                for box in boxes:
                    x_center, y_center, box_width, box_height = box.xywh[0].tolist()
                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)
                    pred_boxes.append([int(box.cls.item()), x1, y1, x2, y2])

                # Compute IoU
                iou_values = []
                matched_gt_boxes = [False] * len(gt_boxes)  # Track matched ground truth boxes
                false_positives = 0  # Track unmatched predicted boxes

                for pred_box in pred_boxes:
                    pred_class, *pred_coords = pred_box
                    max_iou = 0
                    matched_index = -1

                    for idx, gt_box in enumerate(gt_boxes):
                        gt_class, *gt_coords = gt_box
                        if pred_class == gt_class:  # Compare only matching classes
                            iou = calculate_iou(pred_coords, gt_coords)
                            if iou > max_iou:  # Track the highest IoU
                                max_iou = iou
                                matched_index = idx

                    # Record the highest IoU for this predicted box
                    if max_iou >= 0.5:  # Threshold for valid match
                        iou_values.append(max_iou)
                        matched_gt_boxes[matched_index] = True  # Mark the ground truth as matched
                    else:
                        false_positives += 1  # Count as a false positive

                # Count false negatives (unmatched ground truth boxes)
                false_negatives = sum(1 for matched in matched_gt_boxes if not matched)

                if iou_values:
                    avg_iou = sum(iou_values) / len(iou_values)

                # Draw boxes and IoU on the processed image
                processed = draw_boxes_with_iou(processed, pred_boxes, gt_boxes)

        _, buffer = cv2.imencode('.jpg', processed)
        img_str = base64.b64encode(buffer.tobytes()).decode()

        if avg_iou is not None:
            print(f"Average IoU: {avg_iou}")

        response = {'type': 'image', 'data': img_str}
        if avg_iou is not None:
            response['avg_iou'] = avg_iou
            response['false_negatives'] = false_negatives
            response['false_positives'] = false_positives

        return jsonify(response)
    
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        session_id = str(uuid.uuid4())
        temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        file.save(temp_path)
        
        Thread(target=video_processor, 
               args=(temp_path, model, image_size, conf, session_id),
               daemon=True).start()
               
        return jsonify({'type': 'video', 'session_id': session_id})

    return jsonify({'error': 'Unsupported file type'}), 400


@app.route('/get_ground_truth/<path:filename>')
def get_ground_truth(filename):
    label_path = os.path.join('static', 'thermal_dataset', 'labels',
                              os.path.splitext(filename)[0] + '.txt')
    img_path = os.path.join('static', 'thermal_dataset', 'images', filename)

    if not os.path.exists(label_path) or not os.path.exists(img_path):
        return jsonify({'error': 'Label or image file not found'}), 404

    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Read YOLO labels (returns [class_id, x1, y1, x2, y2])
    boxes = read_yolo_label(label_path, width, height)

    # Draw each bounding box
    for class_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode()

    return jsonify({'ground_truth': img_str})

@app.route('/get_examples')
def get_examples():
    example_path = os.path.join('static', 'thermal_dataset', 'images')
    images = [f for f in os.listdir(example_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return jsonify(images)

@app.route('/examples/<path:filename>')
def serve_example(filename):
    return send_from_directory(os.path.join('static', 'thermal_dataset', 'images'), filename)

def read_yolo_label(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append([int(class_id), x1, y1, x2, y2])
    return boxes

def yolo_to_corners(box, width, height):
    """Convert YOLO format to corner coordinates."""
    x_center, y_center, box_width, box_height = box
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def draw_boxes_with_iou(img, pred_boxes, gt_boxes):
    # Draw ground truth boxes in green
    for gt_box in gt_boxes:
        gt_class, x1, y1, x2, y2 = gt_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw predicted boxes in blue and display IoU
    for pred_box in pred_boxes:
        pred_class, x1, y1, x2, y2 = pred_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Compute IoU with ground truth boxes of the same class
        ious = []
        for gt_box in gt_boxes:
            gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            if pred_class == gt_class:
                iou = calculate_iou([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                ious.append(iou)
        if ious:
            max_iou = max(ious)
            cv2.putText(img, f'IoU: {max_iou:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
    return img

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)