import os
import shutil
import random
from PIL import Image

# Set up base paths
base_path = r"C:\Users\theon\OneDrive\Desktop\combine_data"
dataset1_images = os.path.join(base_path, "dataset", "dataset1", "images")
dataset1_labels = os.path.join(base_path, "dataset", "dataset1", "labels")

dataset2_normal = os.path.join(base_path, "dataset", "dataset2", "Normal")
dataset2_pothole = os.path.join(base_path, "dataset", "dataset2", "Pothole")

output_base = os.path.join(base_path, "pothole_combined_yolov8")
images_train = os.path.join(output_base, "images", "train")
images_val = os.path.join(output_base, "images", "val")
labels_train = os.path.join(output_base, "labels", "train")
labels_val = os.path.join(output_base, "labels", "val")

# Create output directories
for folder in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(folder, exist_ok=True)

def yolo_dummy_label(img_path, class_id):
    img = Image.open(img_path)
    w, h = img.size
    x_center = 0.5
    y_center = 0.5
    width = 1.0
    height = 1.0
    return f"{class_id} {x_center} {y_center} {width} {height}\n"

# Collect all images and labels
combined = []

# From dataset1: use existing images and labels
for img_name in os.listdir(dataset1_images):
    img_path = os.path.join(dataset1_images, img_name)
    label_path = os.path.join(dataset1_labels, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    if os.path.exists(label_path):
        combined.append((img_path, label_path))

# From dataset2: create dummy labels (Normal=0, Pothole=1)
for cls, folder in [("0", dataset2_normal), ("1", dataset2_pothole)]:
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        label_txt = yolo_dummy_label(img_path, cls)
        tmp_label_path = os.path.join(base_path, "tmp_" + img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Save the label temporarily
        with open(tmp_label_path, "w") as f:
            f.write(label_txt)

        combined.append((img_path, tmp_label_path))

# Shuffle and split into train and val
random.shuffle(combined)
split_index = int(0.8 * len(combined))
train_set = combined[:split_index]
val_set = combined[split_index:]

def copy_set(file_set, img_out_dir, label_out_dir):
    for img_path, label_path in file_set:
        base_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        shutil.copy(img_path, os.path.join(img_out_dir, base_name))
        shutil.copy(label_path, os.path.join(label_out_dir, label_name))

# Copy training and validation sets
copy_set(train_set, images_train, labels_train)
copy_set(val_set, images_val, labels_val)

# Remove temporary label files
for f in os.listdir(base_path):
    if f.startswith("tmp_") and f.endswith(".txt"):
        os.remove(os.path.join(base_path, f))

print("Dataset preparation complete. Ready for YOLOv8 training.")
