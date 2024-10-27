import os
import json
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from src.model.inference_decryption import decrypt_and_infer
from src.privacy.encryption import aes_cipher
from src.attestation.attestation import Attestation
import config

# Initialize Flask app
app = Flask(__name__, template_folder='src/ui/templates', static_folder='src/static')

# Define upload folder and ensure it exists
UPLOAD_FOLDER = os.path.join(config.BASE_DIR, 'src', 'data', 'input_image')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def landing():
    """Render the landing page."""
    return render_template('landing.html')

@app.route('/upload_page')
def upload_page():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Secure file name and path
    filename = secure_filename(file.filename)
    encrypted_filepath = os.path.join(UPLOAD_FOLDER, filename)
    encrypted_data = aes_cipher.encrypt(file.read())
    with open(encrypted_filepath, 'wb') as f:
        f.write(encrypted_data)

    # Run inference to get disease class without revealing to the user
    prediction = decrypt_and_infer(encrypted_filepath)
    attestation = Attestation().generate_attestation(encrypted_filepath, prediction)
    
    # Save the attestation as JSON
    attestation_file = os.path.join(UPLOAD_FOLDER, f"{filename}_attestation.json")
    with open(attestation_file, 'w') as f:
        json.dump(attestation, f)

    # Provide attestation file as downloadable
    return send_file(attestation_file, as_attachment=True, download_name="attestation.json")

@app.route('/view_result', methods=['POST'])
def view_result():
    # Handle attestation file upload
    if 'attestation' not in request.files:
        return "No attestation file part", 400
    attestation_file = request.files['attestation']
    
    # Load and verify attestation data
    try:
        attestation_data = json.load(attestation_file)
        attestation = Attestation()
        if not attestation.verify_attestation(attestation_data):
            return "Invalid attestation file", 400
    except Exception as e:
        return f"Error processing attestation file: {e}", 400
    
    # Disease mapping based on HAM10000 labels (update keys to match possible values)
    disease_mapping = {
        "0": "Melanocytic nevi",
        "1": "Melanoma",
        "2": "Benign keratosis-like lesions",
        "3": "Basal cell carcinoma",
        "4": "Actinic keratoses and intraepithelial carcinoma",
        "5": "Vascular lesions",
        "6": "Dermatofibroma",
        "7": "No Skin Disease",
        "8": "No Skin Disease",
        "9": "No Skin Disease"
    }
    
    # Retrieve and match the disease class with probable disease
    disease_class = str(attestation_data.get("prediction", "Unknown"))  # Ensure it's a string
    disease_name = disease_mapping.get(disease_class, "Unknown Condition")
    
    # Render the result page with the disease name
    return render_template("result.html", disease_class=disease_class, disease_name=disease_name)


if __name__ == "__main__":
    app.run(debug=True, port=2879)
