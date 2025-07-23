import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# --- Funktioner ---
def polar_to_cartesian_manual(polar_img, output_size=28, center=None, max_radius=None):
    H, W = polar_img.shape  # (theta, radius)
    polar_img = np.flipud(polar_img)  # 🟢 fixar upp/ner-problemet
    if center is None:
        center = (output_size // 2, output_size // 2)
    if max_radius is None:
        max_radius = output_size // 2

    cartesian = np.zeros((output_size, output_size), dtype=np.float32)

    y_indices, x_indices = np.indices((output_size, output_size))
    dx = x_indices - center[0]
    dy = y_indices - center[1]

    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # radianer

    # Skala till [0, 1]
    r_norm = r / max_radius
    theta_norm = (theta + np.pi) / (2 * np.pi)

    # Skala till pixelindex
    r_idx = r_norm * (W - 1)
    theta_idx = theta_norm * (H - 1)

    # Interpolera med bilinjär sampling
    r0 = np.clip(np.floor(r_idx).astype(int), 0, W - 2)
    r1 = r0 + 1
    t0 = np.clip(np.floor(theta_idx).astype(int), 0, H - 2)
    t1 = t0 + 1


    wr = r_idx - r0
    wt = theta_idx - t0

    # Gör bilinjär interpolation i polarbilden
    val = (
        (1 - wr) * (1 - wt) * polar_img[t0, r0] +
        wr * (1 - wt) * polar_img[t0, r1] +
        (1 - wr) * wt * polar_img[t1, r0] +
        wr * wt * polar_img[t1, r1]
    )

    # Maskera utanför max_radius
    cartesian[r <= max_radius] = val[r <= max_radius]

    cartesian = np.fliplr(cartesian)
    return cartesian


def log_radial_filter(shape, alpha=10.0):
    H, W = shape
    r = np.linspace(1.0, 1e-3, W)  # inverterad: 0 (vänster) → 1 (höger)
    weight = 1 / np.log2(1 + alpha * r)
    weight /= weight.max()
    return np.tile(weight, (H, 1))


    weight = 1 / np.log2(1 + alpha * r)
    weight /= weight.max()
    return np.tile(weight, (H, 1))


def polar_transform(image_2d):
    image_cv = (image_2d * 255).astype(np.uint8)
    center = (14, 14)
    max_radius = 14
    polar_image_cv = cv2.warpPolar(
        image_cv,
        (28, 28),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR
    )
    return polar_image_cv.astype(np.float32) / 255.0

def threshold_image(image, threshold=0.65):
    return (image > threshold).astype(np.float32)

def reconstruct_image(pca, components, mean_image):
    return np.dot(components, pca.components_) + pca.mean_

def polar_to_cartesian_image(polar_img):
    polar_img_scaled = (polar_img * 255).astype(np.uint8)
    center = (14, 14)
    max_radius = 14
    cartesian_img = cv2.warpPolar(
        polar_img_scaled,
        (28, 28),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return cartesian_img.astype(np.float32) / 255.0

# --- Ladda 5:or ---
transform = transforms.ToTensor()
mnist_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)

images_2d = []
count = 0
for img, label in mnist_dataset:
    if label == 5:
        images_2d.append(img.squeeze(0).numpy())
        count += 1
        if count >= 5:
            break

# --- Träningsdata: 4 st 5:or ---
train_images = images_2d[:4]
test_image = images_2d[4]  # Ny, oprövad 5:a

# --- Skapa medelbild ---
mean_image_2d = np.mean(train_images, axis=0)

# --- Polar transformering ---
polar_images = [polar_transform(img) for img in train_images]
polar_test_image = polar_transform(test_image)
polar_mean_image = polar_transform(mean_image_2d)

# --- PCA på medelbilden ---
pca = PCA(n_components=5)
pca.fit(polar_mean_image)

# --- PCA på träningsbilder ---
recon_images = []
for polar_img in polar_images:
    transformed = pca.transform(polar_img)
    reconstructed = reconstruct_image(pca, transformed, pca.mean_)
    reconstructed_thresholded = threshold_image(reconstructed, threshold=0.45)
    recon_images.append(reconstructed_thresholded)

# --- PCA på oprövad bild ---
transformed_test = pca.transform(polar_test_image)
reconstructed_test = reconstruct_image(pca, transformed_test, pca.mean_)
radial_filter = log_radial_filter(reconstructed_test.shape, alpha=160.0)
#reconstructed_test_filtered = reconstructed_test * radial_filter

# --- Tröska efter filtrering ---
#reconstructed_test_thresholded = threshold_image(reconstructed_test_filtered, threshold=0.45)

reconstructed_test_thresholded = threshold_image(reconstructed_test, threshold=0.45)

# --- Kartesiska rekonstruktioner ---
recon_images_cartesian = [polar_to_cartesian_manual(img) for img in recon_images]
recon_test_cartesian = polar_to_cartesian_manual(reconstructed_test_thresholded)
recon_test_cartesian = polar_to_cartesian_manual(reconstructed_test)
# --- Visualisering ---
plt.figure(figsize=(18, 12))

for idx in range(4):
    plt.subplot(5, 5, idx*5 + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(f'5:a nr {idx+1} (Kartesisk)')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 2)
    plt.imshow(polar_images[idx], cmap='gray', aspect='auto')
    plt.title('Polar')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 3)
    plt.imshow(recon_images[idx], cmap='gray', aspect='auto')
    plt.title('Återskapad (Polär)')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 4)
    plt.imshow(polar_images[idx] - recon_images[idx], cmap='bwr', aspect='auto')
    plt.title('Skillnad')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 5)
    plt.imshow(recon_images_cartesian[idx], cmap='gray')
    plt.title('Återskapad (Kartesisk)')
    plt.axis('off')

# --- Extra rad: oprövad 5:a ---
plt.subplot(5, 5, 21)
plt.imshow(test_image, cmap='gray')
plt.title('Oprövad 5:a (Kartesisk)')
plt.axis('off')

plt.subplot(5, 5, 22)
plt.imshow(polar_test_image, cmap='gray', aspect='auto')
plt.title('Polar (Test)')
plt.axis('off')

plt.subplot(5, 5, 23)
plt.imshow(reconstructed_test_thresholded, cmap='gray', aspect='auto')
plt.title('Återskapad (Polär)')
plt.axis('off')

plt.subplot(5, 5, 24)
plt.imshow(polar_test_image - reconstructed_test_thresholded, cmap='bwr', aspect='auto')
plt.title('Skillnad (Test)')
plt.axis('off')

plt.subplot(5, 5, 25)
plt.imshow(recon_test_cartesian, cmap='gray')
plt.title('Återskapad (Kartesisk)')
plt.axis('off')

plt.tight_layout()
plt.show()

# --- Visualization ---
plt.figure(figsize=(18, 12))

for idx in range(4):
    plt.subplot(5, 5, idx*5 + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(f'Training 5 number {idx+1} (Cartesian)')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 2)
    plt.imshow(polar_images[idx], cmap='gray', aspect='auto')
    plt.title('Polar image')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 3)
    plt.imshow(recon_images[idx], cmap='gray', aspect='auto')
    plt.title('Reconstructed (Polar)')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 4)
    plt.imshow(polar_images[idx] - recon_images[idx], cmap='bwr', aspect='auto')
    plt.title('Difference')
    plt.axis('off')

    plt.subplot(5, 5, idx*5 + 5)
    plt.imshow(recon_images_cartesian[idx], cmap='gray')
    plt.title('Reconstructed (Cartesian)')
    plt.axis('off')

# --- Extra row: Unseen 5 ---
plt.subplot(5, 5, 21)
plt.imshow(test_image, cmap='gray')
plt.title('Unseen 5 (Cartesian)')
plt.axis('off')

plt.subplot(5, 5, 22)
plt.imshow(polar_test_image, cmap='gray', aspect='auto')
plt.title('Polar image (Test)')
plt.axis('off')

plt.subplot(5, 5, 23)
plt.imshow(reconstructed_test_thresholded, cmap='gray', aspect='auto')
plt.title('Reconstructed (Polar)')
plt.axis('off')

plt.subplot(5, 5, 24)
plt.imshow(polar_test_image - reconstructed_test_thresholded, cmap='bwr', aspect='auto')
plt.title('Difference (Test)')
plt.axis('off')

plt.subplot(5, 5, 25)
plt.imshow(recon_test_cartesian, cmap='gray')
plt.title('Reconstructed (Cartesian)')
plt.axis('off')

plt.tight_layout()
plt.show()
