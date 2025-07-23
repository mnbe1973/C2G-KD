import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Definiera konvolutionella och poolande lager
        self.conv1 = nn.Conv2d(1, 6, 5,padding=2)  # 1 input channel, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 Max Pooling with stride of 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

        # Definiera fullt anslutna lager
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5 from conv2 output dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)            # 10 output classes

    def forward(self, x):
        # Apply convolutions and non-linearities
        x = self.pool(F.sigmoid(self.conv1(x)))  # Convolution -> Activation -> Pooling
        x = self.pool(F.sigmoid(self.conv2(x)))  # Convolution -> Activation -> Pooling
        # Flatten the output from conv2 for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Fully connected layers with ReLU activations
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # Output layer without non-linearity
        x = (self.fc3(x))
        return x

# Example usage
model = LeNet5()
print(model)


# Steg 1: Ladda MNIST-datasetet
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])  # Normalisera datan

train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# Steg 2: Definiera förlustfunktion och optimerare
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # Korsentropiförlust är standard för klassificeringsproblem



# Steg 3: Träningsloop
def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Nollställ gradienterna
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Beräkna gradienter
            optimizer.step()  # Uppdatera vikter

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Steg 4: Funktion för att utvärdera modellen
def evaluate(model, device, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Kör allt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train(model, device, train_loader, optimizer, criterion, epochs=10)
accuracy = evaluate(model, device, test_loader)
print(f'Test Accuracy: {accuracy:.2f}%')

model_path = "teacher_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Teacher model saved to {model_path}")