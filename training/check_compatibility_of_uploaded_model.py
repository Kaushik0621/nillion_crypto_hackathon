
import torch
from PIL import Image
import aivm_client as aic
from torchvision import transforms


MODEL_NAME = "enhanced_model"

image_path = "test_skin_sample/4_Actinic keratoses_3.jpg"  # Path to the input image file


transform = transforms.Compose([
    transforms.Resize((28, 28)),             
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),                    
    transforms.Normalize((0.5,), (1.0,))     
])


try:
    img = Image.open(image_path).convert('RGB')  
    img_tensor = transform(img).unsqueeze(0)  
    print("Image loaded and transformed successfully.")
except Exception as e:
    print(f"Failed to load or transform image: {e}")
    exit()


try:
    encrypted_input = aic.LeNet5Cryptensor(img_tensor)
    print("Image tensor encrypted successfully.")
except Exception as e:
    print(f"Failed to encrypt the image tensor: {e}")
    exit()


try:
    prediction = aic.get_prediction(encrypted_input, MODEL_NAME)
    predicted_class = torch.argmax(prediction).item()  
    print(f"The predicted class for the input image is: {predicted_class}")
except Exception as e:
    print(f"Prediction failed: {e}")

