import cnn
import sys
import os
from PIL import Image
import filters

POOL_SIZE = 3
POOL_STRIDE = 2

def get_files_from_directory(directory):
    """
    @param directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files_only:
        yield f

def process(image_path):
    conv = cnn.ConvolutionNN(image_path, False)
    conv.image_resize(24)
    kernels = [filters.kernel_one_1, filters.kernel_one_2, filters.kernel_one_3]
    #kernels = [filters.kernel_two_1, filters.kernel_two_2, filters.kernel_two_3]
    flat_layer = []
    for kernel in kernels:
        conv.kernel_load(kernel)
        pooled = conv.process(POOL_SIZE, POOL_STRIDE)
        flat_layer.append(pooled)
        print("Pooled output for kernel:", len(flat_layer))
        print(pooled)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            image_path = sys.argv[1]
            for image in get_files_from_directory(image_path):
                print("Processing image:", image)
                process(image_path + "/" + image)
        except FileNotFoundError:
            print(f"Error: File '{image}' not found")
         # Handle the error (set default, raise custom exception, etc.)
        except Image.UnidentifiedImageError:
            print(f"Error: '{image}' is not a valid image file")
        except PermissionError:
            print(f"Error: Permission denied accessing '{image}'")
        except Exception as e:
            print(f"Unexpected error opening image: {e}")
    else:
        print("Please provide the image path as a command-line argument.")