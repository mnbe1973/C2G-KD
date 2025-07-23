import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Definiera träningsbild och testbild (4x4) ---
train_image2 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1,1 , 1, 0]
], dtype=float)
train_image = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1,0 , 0, 0]
], dtype=float)
test_image = np.array([
    [0.9, 1, 0, 0],
    [0.8, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=float)
def reconstruct_image(pca, components, mean_image):
    return np.dot(components, pca.components_) + pca.mean_
# --- PCA per rad ---

pca = PCA(n_components=2)
pca.fit(train_image)

transformed = pca.transform(test_image)
reconstructed_test_image = reconstruct_image(pca, transformed, pca.mean_)
# --- Återskapa bild ---


# --- Visualisering ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(train_image, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Train image")
axes[0].axis('off')

axes[1].imshow(test_image, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Test image (original)")
axes[1].axis('off')

axes[2].imshow(reconstructed_test_image, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Reconstructed test image ")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("PCA-komponentmatris (4x4):\n", pca.components_)
print("\nMedelvärde per kolumn:\n", pca.mean_)