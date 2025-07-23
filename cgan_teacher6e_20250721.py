import torch
import matplotlib.pyplot as plt
import os


# ----- Inställningar -----
num_classes = 10
num_samples = 160  # hur många bilder per klass att visa
image_shape = (28, 28)  # MNIST


# ----- Ladda och samla bilder -----
all_loaded_images = []


for cls in range(num_classes):
    filename = f"generated_images_class_{cls}.pt"
    if not os.path.exists(filename):
        print(f"⚠️ Fil saknas: {filename}")
        continue


    images = torch.load(filename)  # [1600, 1, 28, 28]
    if images.shape[0] < num_samples:
        print(f"⚠️ För få bilder i klass {cls}, hittade {images.shape[0]}")
        continue


    all_loaded_images.append((cls, images[:num_samples]))


# ----- Plotta bilderna -----
fig, axes = plt.subplots(len(all_loaded_images), num_samples, figsize=(num_samples, len(all_loaded_images)*1.5))


for row, (cls, imgs) in enumerate(all_loaded_images):
    for col in range(num_samples):
        axes[row, col].imshow(imgs[col][0], cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_title(f"Klass {cls}", fontsize=8, loc='left')


plt.suptitle("Visning av sparade genererade bilder per klass (.pt)", fontsize=14)
plt.tight_layout()
plt.savefig("replayed_from_saved_images20250721.png")
plt.show()
