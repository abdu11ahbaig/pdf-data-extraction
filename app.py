from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import pytesseract
import cv2
import sys

import numpy as np
import os
import json
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['GRAPH_FOLDER'] = 'graphs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

# Configure Tesseract based on OS
if sys.platform.startswith('win'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif sys.platform.startswith('darwin'):
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PDF to JSON Converter</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <h1>Upload a PDF file</h1>
        <form id="uploadForm" method="post" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="file" class="form-label">Select PDF</label>
                <input type="file" class="form-control" id="file" name="file" accept=".pdf">
            </div>
            <button type="submit" class="btn btn-primary">Upload and Process</button>
        </form>
        <div id="message" class="mt-3"></div>
        <div id="graph" class="mt-3"></div>
    </div>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            const messageDiv = document.getElementById('message');
            if (response.ok) {
                messageDiv.className = 'alert alert-success';
                messageDiv.innerHTML = 'File processed successfully. View JSON: <a href="/view-json/' + result.output_filename + '" target="_blank">View JSON</a>';
                const graphDiv = document.getElementById('graph');
                graphDiv.innerHTML = '<img src="' + result.graph_path + '" alt="Graph" class="img-fluid">';
            } else {
                messageDiv.className = 'alert alert-danger';
                messageDiv.textContent = 'Error: ' + result.error;
            }
            messageDiv.style.display = 'block';
        });
    </script>
</body>
</html>
"""

@app.route('/view-json/<filename>')
def view_json(filename):
    """Serve JSON files in the browser."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    with open(file_path, 'r') as f:
        json_data = f.read()
    return app.response_class(json_data, mimetype='application/json')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        extracted_data = extract_text_from_pdf(file_path)
        output_filename = save_extracted_data_to_file(filename, extracted_data)
        graph_path = generate_graph(filename, extracted_data)
        return jsonify({"message": "File processed successfully", "output_filename": output_filename, "graph_path": graph_path})
    except Exception as e:
        app.logger.error(f"Failed to process file {filename}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    data = []
    checkbox_count = 0

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        page_data = {"page_number": page_num + 1, "text": text, "images": []}

        # Convert page to an image
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect and count checkbox-like elements
        checkbox_count += detect_checkboxes(img)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            ocr_text = pytesseract.image_to_string(image)
            page_data["images"].append({"index": img_index + 1, "ocr_text": ocr_text})

        data.append(page_data)

    return {"pages": data, "checkbox_count": checkbox_count}

def detect_checkboxes(image):
    checkbox_count = 0

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Check if the contour is roughly square and of a reasonable size
        if 0.8 <= aspect_ratio <= 1.2 and 10 <= w <= 30 and 10 <= h <= 30:
            checkbox_count += 1

    return checkbox_count

def save_extracted_data_to_file(filename, data):
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return output_path

def generate_graph(filename, data):
    base_filename = os.path.splitext(filename)[0]
    graph_path = os.path.join(GRAPH_FOLDER, f"{base_filename}.png")

    page_numbers = [page['page_number'] for page in data['pages']]
    checkbox_counts = [data['checkbox_count']] * len(page_numbers)

    # Create a bar graph for checkbox count
    fig, ax = plt.subplots()
    ax.bar(page_numbers, checkbox_counts, color='blue', alpha=0.7)
    ax.set_xlabel('Page Number')
    ax.set_ylabel('Checkbox Count')
    ax.set_title('Checkbox Count per Page')

    plt.savefig(graph_path)
    plt.close(fig)

    return graph_path


def save_extracted_data_to_file(filename, data):
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return output_path




if __name__ == '__main__':
    app.run(debug=True)

# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import json
# import os

# # Configure the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"path_to_tesseract.exe"

# def pdf_to_images(pdf_path):
#     """Converts each page of the PDF to an image."""
#     doc = fitz.open(pdf_path)
#     images = []
#     for page in doc:
#         pix = page.get_pixmap()
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)
#     doc.close()
#     return images

# def ocr_images_to_text(images):
#     """Applies OCR to each image and returns the extracted text."""
#     texts = []
#     for image in images:
#         text = pytesseract.image_to_string(image, lang='eng')
#         texts.append(text)
#     return texts

# def pdf_ocr_to_json(pdf_path):
#     """Converts a PDF to JSON format with OCR extracted text."""
#     images = pdf_to_images(pdf_path)
#     texts = ocr_images_to_text(images)
#     # Creating a dictionary where each page number is a key
#     return json.dumps({f"Page {i+1}": text for i, text in enumerate(texts)}, indent=4)

# # User input for the PDF path
# user_pdf_path = input("Please enter the path to your PDF file: ")

# # Check if the file exists
# if not os.path.isfile(user_pdf_path):
#     print("File does not exist. Please check the path and try again.")
# else:
#     # Process the PDF
#     result = pdf_ocr_to_json(user_pdf_path)
#     print(result)
