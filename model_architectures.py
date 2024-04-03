import os
import shutil
import logging  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm


dataset_dir = "xray/MedicalExpert-I"
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Checking if directories are already populated
if all(os.listdir(directory) for directory in [train_dir, val_dir, test_dir]):
    print("Directories already populated. Skipping processing.")
else:
    # Iterating over each class
    for class_name in ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)

        # Splitting into train, validation, and test
        train, test = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)
        def copy_images(data, destination):
            os.makedirs(os.path.join(destination, class_name), exist_ok=True)
            for image in data:
                shutil.copy(os.path.join(class_dir, image), os.path.join(destination, class_name))

        # Copy images to respective directories
        copy_images(train, train_dir)
        copy_images(val, val_dir)
        copy_images(test, test_dir)

    print("Processing complete.")
    
    
transformation_training = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformation_val = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


training_dataset = ImageFolder(root="dataset/train", transform=transformation_training)
val_dataset = ImageFolder(root="dataset/val", transform=transformation_val)

print(f"Length of Training Dataset : {len(training_dataset)}")
print(f"Length of Validation Dataset : {len(val_dataset)}")

batch_size_gpu = 10
num_workers_gpu = 4

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size_gpu, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_gpu, shuffle=False)

#ResNet-101 model

model = models.resnet101(weights=True)
for param in model.parameters():
    param.requires_grad = False #Freezing all the layers instead of Fully Connected Layer(Linear Layers)
model.fc.requires_grad = True
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(training_dataset.classes))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch : {epoch+1}")
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Using tqdm to create a progress bar
    with tqdm(train_loader, unit="batch") as t:
        for images, labels in t:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update the progress bar description
            t.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(training_dataset)
    train_accuracy = (correct_train / total_train) * 100
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataset)
    val_accuracy = (correct_val / total_val) * 100
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")