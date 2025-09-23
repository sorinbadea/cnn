import cnn
import sys
from PIL import Image
import filters

def main(image_path):
    conv = cnn.ConvolutionNN(image_path)
    conv.image_resize(24)
    kernels = [filters.kernel_one_1, filters.kernel_one_2, filters.kernel_one_3]
    flat_layer = []
    for kernel in kernels:
        conv.kernel_load(kernel)
        pooled = conv.process()
        flat_layer.append(pooled)
        print("Pooled output for kernel:", len(flat_layer))
        print(pooled)
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            main(image_path)
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