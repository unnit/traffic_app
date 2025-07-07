from flask import Flask, request, jsonify, render_template
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
model = YOLO('yolov8n.pt')

# Define class mappings
class_names_dict = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light'
}

# Target classes
target_classes = [1, 2, 3, 5, 7]

@app.route('/')
def index():
    return render_template('index.html')  # Renders the index.html file

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check file size (10MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({'error': 'File size too large. Maximum size is 10MB.'}), 413

    # Check file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    file_extension = '.' + file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG, and WEBP files are allowed.'}), 400

    try:
        # Read the image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not read the image file. Please ensure it is a valid image.'}), 400
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img)
        boxes = results[0].boxes.xywh
        class_ids = results[0].boxes.cls
        confs = results[0].boxes.conf

        # Count objects
        counts = {class_names_dict[c]: (class_ids == c).sum().item() for c in target_classes}
        total_detections = sum(counts.values())

        # Determine time based on detections
        if total_detections == 0:
            time = 0
        elif total_detections <= 5 and total_detections > 0:
            time = 30
        elif total_detections <= 10:
            time = 50
        elif total_detections <= 15:
            time = 60
        else:
            time = 75

        # Plot the image with detections
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()

        for box, cls_id, conf in zip(boxes, class_ids, confs):
            if int(cls_id) < len(class_names_dict):
                label = class_names_dict[int(cls_id)]
            else:
                label = f"Unknown Class {int(cls_id)}"

            x, y, w, h = box
            ax.add_patch(plt.Rectangle(
                (x - w / 2, y - h / 2), w, h,
                linewidth=2, edgecolor='r', facecolor='none')
            )
            ax.text(x - w / 2, y - h / 2, f'{label} ({conf:.2f})', fontsize=12, color='r')

        # Save plot to a buffer
        buf = io.BytesIO()

        # Save the figure with tight bounding box and transparent background
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)

        buf.seek(0)
        plt.close()

        # Encode image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Return JSON response
        return jsonify({
            'counts': counts,
            'total_detections': total_detections,
            'time': time,
            'image': img_base64
        })
    
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
