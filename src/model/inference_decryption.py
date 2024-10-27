# src/model/inference_decryption.py

import os
import tempfile
from src.model.run_inference import run_inference
from src.privacy.encryption import aes_cipher

def decrypt_and_infer(encrypted_filepath):
    """Decrypt an encrypted image file and run inference on the decrypted data."""
    try:
        # Read encrypted data from file
        with open(encrypted_filepath, 'rb') as f:
            encrypted_data = f.read()

        # Decrypt data
        decrypted_data = aes_cipher.decrypt(encrypted_data)

        # Write decrypted data to a temporary file for inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(decrypted_data)
            decrypted_filepath = temp_file.name

        # Run inference on the decrypted image
        prediction = run_inference(decrypted_filepath)

        # Clean up the temporary decrypted file
        os.remove(decrypted_filepath)
        
        return prediction

    except Exception as e:
        return f"Failed during decryption or inference: {e}"
