import os
import shutil
import scipy.io
from tqdm import tqdm

# Paths
root_dir = 'data'
img_dir = os.path.join(root_dir, 'jpg')
label_file = os.path.join(root_dir, 'imagelabels.mat')
setid_file = os.path.join(root_dir, 'setid.mat')

# Output folders
splits = ['train', 'val', 'test']
for split in splits:
    for cls in range(1, 18):  # 17 classes, 1-indexed
        os.makedirs(os.path.join(root_dir, split, f'class_{cls:03}'), exist_ok=True)

# Load .mat files
labels = scipy.io.loadmat(label_file)['labels'][0]
setid = scipy.io.loadmat(setid_file)

# Combine train/val/test splits from parts
train_split = setid['trn1'][0].tolist() + setid['trn2'][0].tolist() + setid['trn3'][0].tolist()
val_split = setid['val1'][0].tolist() + setid['val2'][0].tolist() + setid['val3'][0].tolist()
test_split = setid['tst1'][0].tolist() + setid['tst2'][0].tolist() + setid['tst3'][0].tolist()

split_map = {
    'train': train_split,
    'val': val_split,
    'test': test_split
}

# Copy images into subfolders by class
for split, indices in split_map.items():
    print(f'Preparing {split} set...')
    for i in tqdm(indices):
        img_filename = f'image_{i:04}.jpg'  # Flowers17 images: image_0001.jpg to image_1360.jpg
        label = labels[i - 1]  # labels are 1-indexed
        src = os.path.join(img_dir, img_filename)
        dst = os.path.join(root_dir, split, f'class_{label:03}', img_filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: File {src} not found!")

print('✅ Dataset is ready in SubFolder format.')
