import cnn
import sys
import os
from PIL import Image
import filters

POOL_SIZE = 4
POOL_STRIDE = 3
REDUCED_WIDTH = 20
def get_files_from_directory(directory):
    """
    @param directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files_only:
        yield f

def process(image_path, debug=False):
    """
    process the image loaded from image_path
    @param image_path : image location
    @param debug: enable or disable
    """
    conv = cnn.ConvolutionNN(image_path, debug)
    conv.image_resize(REDUCED_WIDTH)
    flat_layer = []
    for key in filters.kernels_digit_one['filters']:
        conv.kernel_load(filters.kernels_digit_one['filters'][key])
        pooled = conv.process(filters.kernels_digit_one['pool_size'],
                              filters.kernels_digit_one['stride'])
        flat_layer.append(pooled)
        print("Pooled output for kernel:", len(flat_layer))
        print(pooled)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "-f":
                image_path = sys.argv[2]
                process(image_path, True)
            elif sys.argv[1] == "-d":
                image_path = sys.argv[2]
                for image in get_files_from_directory(image_path):
                    print("Processing image:", image)
                    process(image_path + "/" + image, False)
            else:
                print("Usage python main.py -f [image file]")
                print("      python main.py -d [folder_with images]")
        except FileNotFoundError:
            print(f"Error: File '{image_path}' not found")
         # Handle the error (set default, raise custom exception, etc.)
        except Image.UnidentifiedImageError:
            print(f"Error: '{image_path}' is not a valid image file")
        except PermissionError:
            print(f"Error: Permission denied accessing '{image_path}'")
        except Exception as e:
            print(f"Unexpected error opening image: {e}")
    else:
        print("Please provide the image path as a command-line argument.")