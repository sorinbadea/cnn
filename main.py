import cnn
import sys
from PIL import Image
import filters

def main(image_path):
    conv = cnn.ConvolutionNN()
    conv.image_load(image_path)
    conv.kernel_load(filters.kernel_one_1)
    conv.process()
    conv.report()

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