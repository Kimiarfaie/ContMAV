import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
image_dir = 'path-to-images'
label_id_dir = 'path-to-ids'
train_dir = 'train folder'
test_dir = 'test folder'
val_dir = 'val folder'

# Create directories if they don't exist
for split_dir in [train_dir, test_dir, val_dir]:
    os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

# Load train/test split CSV
split_csv = pd.read_csv('path-to-splittingfile-liaci')

# Filter out images marked as 'not_used'
split_csv = split_csv[split_csv['split'] != 'not_used']

# Get list of train images
train_images = split_csv[split_csv['split'] == 'train']['file_name'].tolist()

# Split train images into train and validation (90% train, 10% val)
train_images, val_images = train_test_split(train_images, test_size=0.1, random_state=42)

# Function to copy images and labels
def copy_files(file_list, split_dir):
    for file_name in file_list:
        # Copy image
        image_path = os.path.join(image_dir, file_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(split_dir, 'images', file_name))
        
        # Copy corresponding label ID image (assuming label has the same name as the image)
        label_id_path = os.path.join(label_id_dir, file_name.replace('.jpg', '.png'))
        if os.path.exists(label_id_path):
            shutil.copy(label_id_path, os.path.join(split_dir, 'labels', file_name.replace('.jpg', '.png')))


# Copy train, val, and test data
copy_files(train_images, train_dir)
copy_files(val_images, val_dir)

# Process test set
test_images = split_csv[split_csv['split'] == 'test']['file_name'].tolist()
copy_files(test_images, test_dir)

print("Data split with both label IDs and colored labels completed successfully!")
