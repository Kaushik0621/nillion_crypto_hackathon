# src/model/run_inference.py

import torch
from PIL import Image
import aivm_client as aic
from torchvision import transforms
import config

# Define image transformations based on config settings
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=config.GRAYSCALE_CHANNELS),
    transforms.ToTensor(),
    transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,))
])

def run_inference(image_path):
    """Run inference on the uploaded image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
    except Exception as e:
        return f"Failed to load or transform image: {e}"

    # Encrypt the image tensor for prediction
    try:
        encrypted_input = aic.LeNet5Cryptensor(img_tensor)
    except Exception as e:
        return f"Failed to encrypt the image tensor: {e}"

    # Perform prediction
    try:
        prediction = aic.get_prediction(encrypted_input, config.MODEL_NAME)
        predicted_class = torch.argmax(prediction).item()
        return predicted_class
    except Exception as e:
        return f"Prediction failed: {e}"
