import os
from PIL import Image
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm
import shutil

# Paths
orig_dir = r'C:\Users\HP\OneDrive\retinal\archive\Combined( All types -converted- 746)'
aug_dir = r'C:\Users\HP\OneDrive\retinal\augmented'
final_dir = r'C:\Users\HP\OneDrive\retinal\final_dataset'

# Settings
num_aug_per_image = 2        # 2 augmented copies per image
rotation_limit = 15          # ±15° rotation
brightness_limit = 0.2       # ±20% brightness
contrast_limit = 0.2         # ±20% contrast

# Make folders
os.makedirs(aug_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

# Define augmentations
transform = A.Compose([
    A.Rotate(limit=rotation_limit, interpolation=cv2.INTER_LINEAR,
             border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                               contrast_limit=contrast_limit, p=1.0),
])

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

# Apply augmentations
total_saved = 0
for root, dirs, files in os.walk(orig_dir):
    rel = os.path.relpath(root, orig_dir)
    out_subdir = os.path.join(aug_dir, rel) if rel != '.' else aug_dir
    os.makedirs(out_subdir, exist_ok=True)

    for fname in tqdm(files):
        if not fname.lower().endswith(VALID_EXTS):
            continue
        src_path = os.path.join(root, fname)
        img = np.array(Image.open(src_path).convert('RGB'))
        base = os.path.splitext(fname)[0]

        for i in range(num_aug_per_image):
            aug_img = transform(image=img)['image']
            out_name = f"{base}_aug_{i+1}.png"
            Image.fromarray(aug_img).save(os.path.join(out_subdir, out_name))
            total_saved += 1

# Copy originals + augmented to final folder
for root, dirs, files in os.walk(orig_dir):
    rel = os.path.relpath(root, orig_dir)
    out_subdir = os.path.join(final_dir, rel) if rel != '.' else final_dir
    os.makedirs(out_subdir, exist_ok=True)
    for fname in files:
        if fname.lower().endswith(VALID_EXTS):
            shutil.copy(os.path.join(root, fname), os.path.join(out_subdir, fname))

for root, dirs, files in os.walk(aug_dir):
    rel = os.path.relpath(root, aug_dir)
    out_subdir = os.path.join(final_dir, rel) if rel != '.' else final_dir
    os.makedirs(out_subdir, exist_ok=True)
    for fname in files:
        if fname.lower().endswith(VALID_EXTS):
            shutil.copy(os.path.join(root, fname), os.path.join(out_subdir, fname))

print(f"Done! Total images ready in final folder: {sum([len(files) for r, d, files in os.walk(final_dir)])}")
