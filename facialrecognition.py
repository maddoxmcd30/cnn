import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import os

# Load the dataset
# Dataframes consisting of their respective collections of happy, sad, and surprised faces
# These images must first be mass converted to 2D arrays using OpenCV

trainingSet = pd.DataFrame()
testingSet = pd.DataFrame()

def convert_images_to_arrays(image_folder):
    image_list = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): # Check if the file is an image
            img_path = os.path.join(image_folder, filename)
            img = cv.imread(img_path)
            if img is not None:
                image_list.append(img)
            else:
                print(f"Error reading image: {filename}")
    return np.array(image_list)

def create_new_image_array(image_folder):
#    image_folder = 'archiveDataset/train/happy'
    image_arrays = convert_images_to_arrays(image_folder)
    if image_arrays.size > 0:
        print(f"Successfully converted {len(image_arrays)} images to arrays.")
        #plt.imshow(image_arrays[0])
        #plt.title("Example Image")
        #plt.axis('off')
        #plt.show()
        #print(image_arrays[0].shape)
        # Further processing with image_arrays (e.g., saving to a file)
    else:
        print("No images were converted.")




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Image size (adjust if needed)
IMAGE_SIZE = 48

# Transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale; if RGB, use 3 channels
])

# Datasets
train_dir = 'archiveDataset/train'
test_dir = 'archiveDataset/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1 ),  # If grayscale, change to 1 input channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 emotion classes: happy, sad, surprise
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def training():
    # Initialize model
    model = EmotionCNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    # Testing accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), "model2.pth")


while True:
    response = int(input("What would you like do?\n0: Quit\n"
                         "1: train a new  model\n2:Test specific image\n"))
    match response:
        case 0:
            break
        case 1:
            training()
        case 2:
            model = EmotionCNN().to(device)
            model.load_state_dict(torch.load("model2.pth"))
            img = Image.open("images/henry happy2.jpg").convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

            # Run the model
            output = model(input_tensor)
            prediction = output.argmax(dim=1)

            print("Predicted class:", prediction.item())






