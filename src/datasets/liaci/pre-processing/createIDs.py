import os
import numpy as np
from PIL import Image

# Define the mapping from RGB to label IDs, this is specific for OW LIACi dataset, if you want bilge_keel - orange and marine_growth - green included, change 0 to index
rgb_to_label = {
    (0, 0, 0): 0,  # Null
    (255, 255, 255): 1, #see_chest_grating - white
    (255, 0, 0): 2, #paint_peel - Red
    (64, 224, 208): 3, #overboard_valves - turquoise
    (254, 193, 203): 4,  # defect - light pink
    (255, 255, 0): 5,  # corrosion - yellow
    (128, 0, 128): 6, # propeller - purple
    (0, 255, 255): 7,  # Anod - cyan
    (255, 165, 0): 0,    # bilge_keel - orange
    (0, 128, 0): 0, #marine_growth - green
    (0, 0, 255): 8, #Shiphull - blue
    
}

def convert_rgb_to_label(image, rgb_to_label):
    """
    Convert an RGB segmented image to a label ID image based on the rgb_to_label mapping.
    """
    # Convert the image to a numpy array
    image_np = np.array(image)
    unique_colors = np.unique(image_np.reshape(-1, image_np.shape[-1]), axis=0)

    # Initialize a blank label image with the same width and height
    label_image = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    # Apply the mapping from RGB to label ID
    for rgb, label in rgb_to_label.items():

        mask = np.all(image_np == rgb, axis=-1)  # Check where the RGB values match

        label_image[mask] = label  # Assign label ID to those positions
    
    return label_image

# Input and output folder paths
input_folder = "path-to-segmentation-masks"
output_folder = "path-to-savedfolder"  

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Process only PNG images (adjust as needed)
        # Open the RGB segmented image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Convert the RGB image to a label ID image
        label_image = convert_rgb_to_label(image, rgb_to_label)

        # Save the label ID image
        output_path = os.path.join(output_folder, filename)
        label_image_pil = Image.fromarray(label_image)
        label_image_pil.save(output_path)

        print(f"Processed and saved {filename} as label ID image.")
