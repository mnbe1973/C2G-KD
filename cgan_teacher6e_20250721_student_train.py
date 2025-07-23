import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# ----- Modell -----
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# ----- Inställningar -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
num_samples_per_class = 160  # antal bilder att använda från varje .pt-fil

# ----- Ladda .pt-filer och skapa dataset -----
images_list = []
labels_list = []

for cls in range(num_classes):
    filename = f"generated_images_class_{cls}.pt"
    if not os.path.exists(filename):
        print(f"⚠️ Fil saknas: {filename}")
        continue

    images = torch.load(filename)  # [1600, 1, 28, 28]
    images = images[:num_samples_per_class]  # Begränsa om behövs

    labels = torch.full((images.size(0),), cls, dtype=torch.long)  # Skapa etiketter för klassen

    images_list.append(images)
    labels_list.append(labels)

# Slå ihop alla bilder och etiketter
X_train = torch.cat(images_list)
y_train = torch.cat(labels_list)

print(f"Totalt antal träningsbilder: {X_train.shape[0]}")

# ----- Bygg DataLoader -----
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ----- Studentmodell och träning -----
student = LeNet5().to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs_student = 20
for epoch in range(num_epochs_student):
    running_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = student(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs_student}, Loss: {running_loss/len(dataloader):.4f}")

# ----- Utvärdera -----
from torchvision import datasets, transforms

mnist_test = datasets.MNIST(root=".", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)

student.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = student(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"\nStudentmodellens noggrannhet på riktig MNIST-testdata: {100 * correct / total:.2f}%")
