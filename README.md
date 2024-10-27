
# **Nillion Crypto Hackathon - House Your Skin**

## **Overview**
This project demonstrates privacy-preserving image classification, developed for the Nillion Crypto Hackathon. It utilizes advanced techniques like encryption, zero-knowledge proofs (ZKP), multi-party computation (MPC), and attestations to ensure secure and private classification of skin disease images. Decentralized storage through IPFS provides tamper-proof records of each prediction.

## **Features**
- **Secure Image Upload**: Encrypts uploaded images to protect user privacy.
- **Secure Inference**: Performs classification on encrypted images using a pre-trained LeNet5 model.
- **Attestation**: Generates an attestation for each prediction, ensuring proof of results.
- **Decentralized Storage**: Stores attestations on IPFS for secure, immutable records.
- **Zero-Knowledge Proofs (ZKP)**: Verifies prediction integrity without exposing data.
- **Multi-Party Computation (MPC)**: Aggregates data securely using secret-sharing methods.

---

## **Folder Structure**
```
nillion_crypto_hackathon/
│
└── src/
    ├── attestation/              # Attestation and decentralized storage management
    ├── data/                     # Contains input images and test data samples
    ├── federated/                # Placeholder for federated learning (currently empty)
    ├── model/                    # Inference and decryption methods
    ├── privacy/                  # Implements encryption, MPC, and ZKP functions
    ├── static/                   # Static files for UI (if any)
    ├── ui/                       # HTML templates and UI configuration
    ├── training/                 # Scripts for model management and compatibility checking
    ├── utils/                    # Utility functions (if any)
    ├── app.py                    # Main Flask application
    ├── config.py                 # Configuration for paths, encryption settings, etc.
    └── README.md                 # Project documentation
```

---

## **Installation and Setup**

### **Prerequisites**
- **Python 3.8+**
- **pip** for Python package management
- **IPFS** (InterPlanetary File System) for decentralized storage
- **Nillion AIVM** for secure inference
- **Git** (optional, for cloning the repository)

### **Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd nillion_crypto_hackathon
```

### **Step 2: Set Up and Activate a Virtual Environment**
It’s recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv env
source env/bin/activate  # For Linux/macOS
env\Scripts\activate     # For Windows
```

### **Step 3: Install Dependencies**
Install all required Python packages.

```bash
pip install -r requirements.txt
```

### **Step 4: Configure IPFS**
Install and configure IPFS for decentralized attestation storage.

1. **Download IPFS**: Follow the instructions at [IPFS Installation](https://docs.ipfs.io/install/).
2. **Initialize IPFS**:
   ```bash
   ipfs init
   ```
3. **Run the IPFS Daemon**:
   ```bash
   ipfs daemon
   ```
   IPFS should now be running on `http://127.0.0.1:5001`.

### **Step 5: Update Configuration**
Edit `config.py` to customize encryption keys, model name, and other settings:

- **AES Key**: Replace the placeholder AES key for production use.
- **Model Name**: Set `MODEL_NAME` to identify the model on the server.
  
**Example**:
```python
# config.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (28, 28)
GRAYSCALE_CHANNELS = 1
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 1.0
MODEL_NAME = "enhanced_model"
AES_KEY = b'SecureAESKeyHere123SecureAESKeyHere123'
```

### **Step 6: Run the Application**

To run the application fully, you will need to start three separate terminals:

#### **Terminal 1: Start the Nillion AIVM**
Start the Nillion AI Virtual Machine (AIVM) required for secure inference:

```bash
# Start the Nillion AIVM (replace with the actual command to start Nillion AIVM)
nillion aivm start
```

Ensure that the AIVM is running and accessible.

#### **Terminal 2: Start the IPFS Daemon**
IPFS is used for decentralized storage of attestation files. Start the IPFS daemon as follows:

```bash
# Start IPFS Daemon
ipfs daemon
```

The IPFS daemon will run on `http://127.0.0.1:5001` by default.

#### **Terminal 3: Start the Flask Application**
Finally, start the Flask application to enable the web interface for uploading images, generating predictions, and verifying results:

```bash
# Start the Flask application
python app.py
```

The app will be available at `http://127.0.0.1:2879`.

### **Summary of Terminals Needed**
1. **Terminal 1**: Nillion AIVM for secure inference
2. **Terminal 2**: IPFS Daemon for decentralized attestation storage
3. **Terminal 3**: Flask application (`app.py`) for the user interface

Ensure all three services are running before interacting with the application.

---

## **Usage**

### **1. Upload an Image**
   - Go to the upload page (`http://127.0.0.1:2879/upload_page`).
   - Select an image file from `src/data/input_image`.
   - The image is encrypted, and a prediction is made based on the model’s classification.

### **2. Download Attestation**
   - After prediction, you will receive an attestation file (JSON format) which records the prediction details and can be downloaded.

### **3. Verify Prediction**
   - Go to the verification page (`/view_result`) and upload the attestation file to view and verify the classification result.

---

## **Detailed Component Descriptions**

### **Attestation**
- **Files**:
  - `attestation.py`: Manages the creation and verification of attestations.
  - `storage.py`: Handles the uploading of attestations to IPFS.
- **Purpose**: Ensures each prediction is recorded and verifiable in a tamper-proof manner using decentralized storage.

### **Model**
- **Files**:
  - `inference_decryption.py`: Decrypts encrypted images and runs inference.
  - `run_inference.py`: Handles the image loading, transformation, encryption, and prediction for skin classification.
- **Purpose**: Facilitates secure inference on encrypted data to protect user privacy.

### **Privacy**
- **Files**:
  - `encryption.py`: Provides AES encryption and decryption methods for secure file handling.
  - `mpc_handler.py`: Implements MPC for secure aggregation using secret sharing.
  - `zk_proof.py`: Demonstrates a basic zero-knowledge proof to verify prediction integrity.
- **Purpose**: Ensures secure data handling through encryption, zero-knowledge proofs, and multi-party computation.

### **Training**
- **Files**:
  - `check_compatibility_of_uploaded_model.py`: Tests the full workflow, including image loading, encryption, and prediction.
  - `model_upload.py`: Uploads the LeNet5 model to the server for inference.
  - `check_models_in_nillion_server.py`: Lists supported models on the `aivm_client` server.
- **Purpose**: Manages and tests model compatibility and uploads the pre-trained model.

---

## **Key Technologies**
- **Flask**: Serves as the backend framework for routing and rendering pages.
- **IPFS**: Used for decentralized, tamper-proof storage of attestation files.
- **AES Encryption**: Provides secure, private storage and handling of images.
- **Zero-Knowledge Proofs (ZKP)**: Allows verification of prediction authenticity without exposing sensitive data.
- **Multi-Party Computation (MPC)**: Aggregates data securely to enhance privacy.

---

## **Security Considerations**
- Replace the hardcoded `AES_KEY` in `config.py` with a secure key management service for production use.
- Ensure compliance with data privacy regulations when using IPFS and other decentralized storage solutions.

---

## **Future Enhancements**
- **Federated Learning**: Utilize the `federated` folder to implement decentralized model training on local data for enhanced privacy.
- **Advanced ZKP**: Expand zero-knowledge proof functionality for more complex verification scenarios.
- **Enhanced Security**: Integrate a secure key management service to replace hardcoded keys.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---
