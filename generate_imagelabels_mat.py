# generate_imagelabels_mat.py
import numpy as np
import scipy.io

# Each class has 80 images, 17 classes → total 1360 images
labels = []

for class_id in range(1, 18):  # class IDs from 1 to 17
    labels.extend([class_id] * 80)

labels = np.array(labels, dtype=np.uint8)

# Save to .mat file
scipy.io.savemat('data/imagelabels.mat', {'labels': labels})

print("✅ imagelabels.mat created with 1360 labels (17 classes × 80 images)")
