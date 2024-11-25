import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Dataset pathz
base_dir = "./3-Dataset-Structure"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Dataset and data loaders
train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(base_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Classes: {train_dataset.classes}")

# Model setup
model = models.resnet18(pretrained=True)  # Pretrained ResNet18
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust output layer
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
model_save_path = "baseline_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")