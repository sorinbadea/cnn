from PIL import Image
import numpy as np
import sys

# Load the image
image = Image.open(sys.argv[1])

# Convert to grayscale (0-255 values)
grayscale = image.convert('L')
array = np.array(grayscale)

# Or keep RGB channels (3D array with values 0-255)
#rgb_array = np.array(image)

print(f"Shape: {array.shape}")  # (height, width) or (height, width, channels)
print(f"Number of rows: {array.shape[0]}")
print(f"Number of columns: {array.shape[1]}")
print(f"Data type: {array.dtype}")