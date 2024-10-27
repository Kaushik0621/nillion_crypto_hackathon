# src/attestation/storage.py

import subprocess
import json

class DecentralizedStorage:
    def upload_attestation(self, attestation_data: dict) -> str:
        """Upload attestation data to IPFS and return the local IPFS URL."""
        # Convert the attestation data to JSON and save it to a temporary file
        attestation_json = json.dumps(attestation_data)
        
        # Save the JSON data to a file
        with open("attestation.json", "w") as f:
            f.write(attestation_json)
        
        # Use the IPFS CLI to add the file and capture the output
        result = subprocess.run(["ipfs", "add", "attestation.json"], capture_output=True, text=True)
        ipfs_hash = result.stdout.split()[1]  # Extract the hash from the output
        
        # Return the local IPFS gateway URL
        return f"http://127.0.0.1:8080/ipfs/{ipfs_hash}"

    def retrieve_attestation(self, ipfs_hash: str) -> dict:
        """Retrieve attestation data from IPFS using IPFS CLI."""
        result = subprocess.run(["ipfs", "cat", ipfs_hash], capture_output=True, text=True)
        attestation_json = result.stdout
        return json.loads(attestation_json)
