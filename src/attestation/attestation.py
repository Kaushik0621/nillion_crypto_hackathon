# src/attestation/attestation.py

import hashlib
import time
import json
import config

class Attestation:
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name

    def generate_attestation(self, image_path: str, prediction: str) -> dict:
        """Generate a unique attestation for an inference."""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Unique identifier and timestamp for attestation
        timestamp = time.time()
        unique_id = hashlib.sha256(f"{self.model_name}_{timestamp}".encode()).hexdigest()

        # Generate a hash of the image, model, and prediction
        attestation_data = {
            "unique_id": unique_id,
            "model_name": self.model_name,
            "prediction": prediction,
            "timestamp": timestamp,
            "image_hash": hashlib.sha256(image_data).hexdigest()
        }
        
        # Create the final attestation by hashing the attestation data
        attestation_hash = hashlib.sha256(json.dumps(attestation_data).encode()).hexdigest()
        attestation_data["attestation_hash"] = attestation_hash
        
        return attestation_data

    def verify_attestation(self, attestation_data: dict) -> bool:
        """Verify an attestation to ensure it has not been tampered with."""
        # Recreate the hash from the attestation data
        attestation_data_copy = attestation_data.copy()
        provided_hash = attestation_data_copy.pop("attestation_hash", None)
        recalculated_hash = hashlib.sha256(json.dumps(attestation_data_copy).encode()).hexdigest()

        # Verification: Check if the provided hash matches the recalculated hash
        return provided_hash == recalculated_hash
