from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# Directory to save uploaded and processed images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image. Ensure it's a valid grayscale image.")
    
    # Step 1: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)
    
    # Step 2: Smooth image to suppress noise
    blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    
    # Step 3: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, -10)
    
    # Step 4: Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 5: Keep largest components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + np.argmax(areas)
    vessel_mask = np.uint8(labels == max_label) * 255
    
    # Save processed images with unique filenames
    unique_id = str(uuid.uuid4())
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{unique_id}.png')
    morph_path = os.path.join(app.config['UPLOAD_FOLDER'], f'morph_{unique_id}.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'mask_{unique_id}.png')
    
    cv2.imwrite(original_path, img)
    cv2.imwrite(morph_path, morph)
    cv2.imwrite(mask_path, vessel_mask)
    
    return {
        'original': original_path,
        'morph': morph_path,
        'vessel_mask': mask_path
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                # Process image
                result = process_image(file_path)
                return render_template('index.html', images=result)
            except Exception as e:
                return render_template('index.html', error=str(e))
        else:
            return render_template('index.html', error='Invalid file format. Use JPG or PNG.')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)