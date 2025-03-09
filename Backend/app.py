from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load YOLOS model and processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
device = torch.device("cpu")
model.to(device)

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Get the image from the request
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])  # Decode base64 image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Process the image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the results
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    # Format the results
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": score.item(),
            "box": box.tolist()
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)