
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_data_dir = './Data'

# Load metadata
skin_df = pd.read_csv(os.path.join(base_data_dir, 'HAM10000_metadata.csv'))

# Create a dictionary of image paths
imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(base_data_dir, 'HAM10000_images_part_1', '*.jpg')) +
             glob(os.path.join(base_data_dir, 'HAM10000_images_part_2', '*.jpg'))
}

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),              
    transforms.Grayscale(num_output_channels=1),  
    transforms.RandomHorizontalFlip(),        
    transforms.RandomRotation(10),           
    transforms.ToTensor(),                    
    transforms.Normalize((0.5,), (1.0,))     
])

def load_and_transform_image(path):
    image = Image.open(path)
    image = transform(np.array(image))
    return image


skin_df['image'] = skin_df['path'].map(load_and_transform_image)


features = skin_df['image'].tolist()
targets = skin_df['cell_type_idx'].tolist()


X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.1, random_state=42, stratify=targets
)


train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)  
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


network = LeNet5().to(device)
optimizer = optim.Adam(network.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = network(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test():
    network.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy().flatten())
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy
    ))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(lesion_type_dict.values()))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix on Test Set")
    plt.show()


N_EPOCHS = 10  
for epoch in range(1, N_EPOCHS + 1):
    train(epoch)
    test()


torch.save(network.state_dict(), "lenet5_model.pth")


loaded_network = LeNet5().to(device)
loaded_network.load_state_dict(torch.load("lenet5_model.pth", map_location=device), strict=False)
print("Model loaded successfully.")
