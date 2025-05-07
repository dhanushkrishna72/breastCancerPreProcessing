import os
import cv2
import numpy as np

# Set paths
input_train_dir = 'BreastCancerMRIs/train'
input_val_dir = 'BreastCancerMRIs/validation'

output_train_dir = 'BreastCancerMRIs/processed/train'
output_val_dir = 'BreastCancerMRIs/processed/validation'

# Desired output image size
IMG_SIZE = (224, 224)

# Create output folders
for dir_path in [output_train_dir, output_val_dir]:
    os.makedirs(os.path.join(dir_path, 'healthy'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'sick'), exist_ok=True)

# Function to load, preprocess and save images
def preprocess_and_save(input_dir, output_dir):
    for condition in ['healthy', 'sick']:
        input_path = os.path.join(input_dir, condition)
        output_path = os.path.join(output_dir, condition)
        
        for img_name in os.listdir(input_path):
            img_input_path = os.path.join(input_path, img_name)
            img_output_path = os.path.join(output_path, img_name)

            img = cv2.imread(img_input_path)

            if img is None:
                continue  # Skip bad images
            
            # Resize image
            img = cv2.resize(img, IMG_SIZE)

            # Convert to grayscale (optional)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values (0 to 255)
            img = (img * (255.0 / img.max())).astype(np.uint8)

            # Save the preprocessed image
            cv2.imwrite(img_output_path, img)

# Preprocess and save train and validation images
preprocess_and_save(input_train_dir, output_train_dir)
preprocess_and_save(input_val_dir, output_val_dir)

print("Preprocessing and saving completed!")
