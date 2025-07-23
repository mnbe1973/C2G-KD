import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

def generate_digit_image(digit, size=28, thickness=2):
    segment_map = {
        0: [1, 1, 1, 0, 1, 1, 1],
        1: [0, 0, 1, 0, 0, 1, 0],
        2: [1, 0, 1, 1, 1, 0, 1],
        3: [1, 0, 1, 1, 0, 1, 1],
        4: [0, 1, 1, 1, 0, 1, 0],
        5: [1, 1, 0, 1, 0, 1, 1],
        6: [1, 1, 0, 1, 1, 1, 1],
        7: [1, 0, 1, 0, 0, 1, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1],
    }
    segments = segment_map.get(digit, [0]*7)
    img = np.zeros((size, size), dtype=np.float32)
    margin = size // 8
    w = thickness
    if segments[0]: img[margin:margin+w, margin:size-margin] = 1.0
    if segments[1]: img[margin:size//2, margin:margin+w] = 1.0
    if segments[2]: img[margin:size//2, size-margin-w:size-margin] = 1.0
    if segments[3]: img[size//2 - w//2:size//2 + w//2 + 1, margin:size-margin] = 1.0
    if segments[4]: img[size//2:size-margin, margin:margin+w] = 1.0
    if segments[5]: img[size//2:size-margin, size-margin-w:size-margin] = 1.0
    if segments[6]: img[size-margin-w:size-margin, margin:size-margin] = 1.0
    return img

def cartesian_to_polar(img):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    maxRadius = min(center[0], center[1])
    polar_img = cv2.warpPolar(img, (img.shape[1], img.shape[0]), center, maxRadius, cv2.WARP_POLAR_LINEAR)
    return polar_img

def polar_to_cartesian(polar_img):
    center = (polar_img.shape[1] // 2, polar_img.shape[0] // 2)
    maxRadius = min(center[0], center[1])
    cartesian_img = cv2.warpPolar(polar_img, (polar_img.shape[1], polar_img.shape[0]), center, maxRadius, cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    return cartesian_img

def shift_pca_vectors(pca_result, shift=14):
    return np.roll(pca_result, shift=shift, axis=0)

# Generera digit och PCA
digit = 9
img = generate_digit_image(digit)
polar_img = cartesian_to_polar(img)
pca = PCA(n_components=5)
pca_result = pca.fit_transform(polar_img)

# Skifta PCA-komponenter 180°
shifted_pca_result = shift_pca_vectors(pca_result, shift=14)

# Återskapa polarbild från skiftad PCA
reconstructed_polar = shifted_pca_result @ pca.components_ + pca.mean_

# Konvertera tillbaka till kartesiskt
reconstructed_cartesian = polar_to_cartesian(reconstructed_polar.astype(np.float32))

# Visa original och rekonstruerad bild
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Digit')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_polar, cmap='gray', aspect='auto')
plt.title('Polar after PCA shift')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_cartesian, cmap='gray')
plt.title('Reconstructed Digit (Shifted)')

plt.tight_layout()
plt.show()
