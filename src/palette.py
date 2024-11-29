import numpy as np
from PIL import Image

# This script is for colorizing the mask and visualization of the predictions, note if you want to go to OW for liaci, as some classes are set to unkown, you have to change the palette.

CityScpates_palette = [
    0,0,0,
    128,64,128,
    244,35,232,
    70,70,70,
    102,102,156,
    190,153,153,
    153,153,153,
    250,170,30,
    220,220,0,
    107,142,35,
    152,251,152,
    70,130,180,
    220,20,60,
    255,0,0,
    0,0,142,
    0,0,70,
    0,60,100,
    0,80,100,
    0,0,230,
    119,11,32
    ]

LIACi_palette = [0, 0, 0, 255, 255, 255, 255, 0, 0, 64, 224, 208, 254, 193, 203, 255, 255, 0, 128, 0, 128, 0, 255, 255, 255, 165, 0, 0, 128, 0, 0, 0, 255]
def get_liaci_palette():
    # Define color mapping for LIACi classes
    palette = np.array(LIACi_palette, dtype=np.uint8).reshape(-1, 3)
    return palette


def get_cityscapes_palette():
    """Returns the Cityscapes palette as a numpy array for visualization."""
    palette = np.array(CityScpates_palette, dtype=np.uint8).reshape(-1, 3)
    return palette

def colorize_mask(mask, output, palette):
    """Convert a mask to RGB using the provided palette, mapping 255 to black."""
    # mask = mask.copy()
    mask += 1 # cause masks are now loaded such as 0 when this is called
    mask[mask==256] = 0 # setting back the 255 (+1) index back to 0
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette.flatten())
    colorized_mask = new_mask.convert('RGB')

    # Set back to white if needed after converting to RGB
    colorized_mask = np.array(colorized_mask)

    output += 1
    output[mask==0] = 0
    new_output = Image.fromarray(output.astype(np.uint8)).convert('P')
    new_output.putpalette(palette.flatten())
    colorized_output = new_output.convert('RGB')
    colorized_output = np.array(colorized_output)
    return Image.fromarray(colorized_mask), Image.fromarray(colorized_output)