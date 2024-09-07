from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import easyocr
import cv2
import numpy as np
from matplotlib import pyplot as plt

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 260:
            cv2.drawContours(opening, [c], -1, 0, -1)

    result = cv2.bitwise_xor(thresh, opening)

    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_result.jpg')
    cv2.imwrite(processed_image_path, result)
    return processed_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_image_path = process_image(filepath)

        # Perform OCR using EasyOCR
        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed_image_path)
        high_confidence_results = [detection for detection in result if detection[2] > 0]

        extracted_text = ' '.join([detection[1] for detection in high_confidence_results])
        bounding_boxes = [{'bounding_box': detection[0], 'text': detection[1]} for detection in high_confidence_results]

        # Highlight detected text with bounding boxes on the image
        image = cv2.imread(filepath)
        for detection in high_confidence_results:
            bounding_box, text, confidence = detection
            top_left = tuple(map(int, bounding_box[0]))
            bottom_right = tuple(map(int, bounding_box[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        result_highlighted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_highlighted.jpg')
        cv2.imwrite(result_highlighted_path, image)

        # Save the image with highlights
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.savefig(result_highlighted_path) 
        plt.close()

        return render_template('result.html', extracted_text=extracted_text, bounding_boxes=bounding_boxes, filepath='uploads/result_highlighted.jpg')

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)

