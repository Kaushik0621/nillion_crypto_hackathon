# config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image transformation settings
IMAGE_SIZE = (28, 28)
GRAYSCALE_CHANNELS = 1
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 1.0

# Model identifier for inference
MODEL_NAME = "enhanced_model"

# Persistent AES key (for testing only; replace with secure key management in production)
AES_KEY = b'This is a key123This is a key123'  # 32 bytes for AES-256
