import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import os

# ----- 1. Ladda lärarmodellen -----
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

teacher = LeNet5()
teacher.load_state_dict(torch.load("teacher_model.pth", map_location=torch.device('cpu')))
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# ----- 2. Conditional Convolutional Generator -----
class ConditionalConvGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalConvGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim + num_classes, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 128, 7, 7)
        x = self.net(x)
        return x

# ----- Elastic transformation för augmentation -----
def elastic_transform(image, alpha=34.0, sigma=4.0):
    image_np = image.squeeze(0).cpu().numpy()
    random_state = np.random.RandomState(None)
    shape = image_np.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(image_np, indices, order=1, mode='reflect')
    distorted = distorted.reshape(shape)
    return torch.from_numpy(distorted).unsqueeze(0).to(image.device)

# ----- PCA-subrymder per klass (Torch-version) -----
def build_classwise_pca_subspaces(n_components=2):
    pca_dict = {}
    mnist_train = datasets.MNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(mnist_train, batch_size=100, shuffle=False)
    images, labels = next(iter(loader))
    for cls in range(10):
        cls_imgs = images[labels == cls].view(-1, 28*28)
        cls_imgs = cls_imgs[:2]
        pca = PCA(n_components=n_components)
        pca.fit(cls_imgs.numpy())
        mean = torch.tensor(pca.mean_, dtype=torch.float32)
        components = torch.tensor(pca.components_, dtype=torch.float32)
        pca_dict[cls] = (mean, components)
    return pca_dict

def pca_projection_loss(images, labels, pca_subspaces):
    images_flat = images.view(images.size(0), -1)
    losses = []
    for cls in range(10):
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        mean, components = pca_subspaces[cls]
        x = images_flat[mask]
        x_centered = x - mean.to(x.device)
        proj = x_centered @ components.T.to(x.device) @ components.to(x.device)
        recon = proj + mean.to(x.device)
        loss = (x - recon).pow(2).sum(dim=1).mean()
        losses.append(loss)
    return torch.stack(losses).mean()

def pca_diversity_loss(images, labels, pca_dict):
    images_flat = images.view(images.size(0), -1)
    loss = 0
    for cls in range(10):
        cls_mask = (labels == cls)
        if cls_mask.sum() < 2: continue
        mean, components = pca_dict[cls]
        x = images_flat[cls_mask] - mean.to(images.device)
        proj = x @ components.T.to(images.device)
        cov = torch.cov(proj.T)
        diversity = -torch.trace(cov)
        loss += diversity
    return loss / 10

pca_subspaces = build_classwise_pca_subspaces()

latent_dim = 100
num_epochs = 3000
lr = 0.001
batch_size = 640
num_samples_per_class = 1600

alpha = 1.0  # vikt för PCA-förlust
beta = 0.1   # vikt för diversitetsförlust

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
generator = ConditionalConvGenerator(latent_dim).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

generated_data = []
all_generated_images = []

for target_class in range(10):
    print(f"Training generator for class {target_class}...")
    for epoch in range(num_epochs):
        z = torch.randn(batch_size, latent_dim).to(device) * torch.rand(1).item() * 2
        labels = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
        fake_images = generator(z, labels)
        outputs = teacher(fake_images)
        probs = F.log_softmax(outputs, dim=1)
        loss_distill = -probs[:, target_class].mean()
        loss_pca = pca_projection_loss(fake_images, labels, pca_subspaces)
        loss_div = pca_diversity_loss(fake_images, labels, pca_subspaces)
        loss = loss_distill + alpha * loss_pca + beta * loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Class {target_class} - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(generator.state_dict(), f"generator_class_{target_class}.pth")

    z = torch.randn(num_samples_per_class, latent_dim).to(device)
    labels = torch.full((num_samples_per_class,), target_class, dtype=torch.long).to(device)
    fake_images = generator(z, labels).detach()
    fake_images = torch.stack([elastic_transform(img) for img in fake_images]).cpu()
    torch.save(fake_images, f"generated_images_class_{target_class}.pt")

    all_generated_images.append((target_class, fake_images))
    generated_data.append((fake_images, torch.full((num_samples_per_class,), target_class)))

#Visualisera variation för en klass efter träning
example_class = 3
generator.load_state_dict(torch.load(f"generator_class_{example_class}.pth"))
generator.eval()
z = torch.randn(16, latent_dim).to(device)
labels = torch.full((16,), example_class, dtype=torch.long).to(device)
with torch.no_grad():
    varied_images = generator(z, labels).cpu()

fig, axes = plt.subplots(1, 16, figsize=(12, 1.5))
for i in range(16):
    axes[i].imshow(varied_images[i][0], cmap='gray')
    axes[i].axis('off')
plt.suptitle(f"Variation från sparad pth för klass {example_class}")
plt.tight_layout()
plt.show()

# Studentträning
print("\nTraining student model on synthetic data...")
images, labels = zip(*generated_data)
images = torch.cat(images)
labels = torch.cat(labels)

dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

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

print(f"\nStudent accuracy on real MNIST test data: {100 * correct / total:.2f}%")
