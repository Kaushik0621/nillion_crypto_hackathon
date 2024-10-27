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

AES_KEY = b'This is a production key which is used to run the application'  # 32 bytes for AES-256
