import cnn
import sys
import os
from PIL import Image
import filters
from database import DataBaseInterface

REDUCED_WIDTH = 24

def usage():
    print("Usage python main.py -f [image file]")
    print("      python main.py -d [folder_with images]")

def get_files_from_directory(directory):
    """
    @param: directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files_only:
        yield f

def process(image_path, verbosity, db=None):
    """
    Process the image loaded from image_path;
    Save for each kernel shape the results in a database;
    @param image_path : image location
    @param verbosity: enable verbosity
    @param: db: database interface
    """
    conv = cnn.ConvolutionNN(image_path, verbosity)
    conv.image_resize(REDUCED_WIDTH)
    conv.grayscale()
    kernel_hash = filters.kernels_digit_one['filters']
    for (key, i) in zip (kernel_hash, range(len(kernel_hash))):
        conv.kernel_load(kernel_hash[key])
        pooled = conv.process(filters.kernels_digit_one['pool_size'],
                              filters.kernels_digit_one['stride'])
        print("Pooled output for kernel:", i)
        print(pooled)
        # save in database
        if db:
            db.insert_data(key, pooled.tolist())
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "-f":
                """
                single image processing mode
                """
                image_path = sys.argv[2]
                process(image_path, True)

            elif sys.argv[1] == "-d":
                image_path = sys.argv[2]
                """
                training mode, update a table in a databse for each kernel shape
                """
                db = DataBaseInterface('localhost','myapp','postgres','password',5432)
                for key in filters.kernels_digit_one['filters']:
                    db.create_table(key)
                """
                start processing all images in the specified folder
                """
                for image in get_files_from_directory(image_path):
                    print("Processing image:", image)
                    process(image_path + "/" + image, False, db)

            else:
                usage()

            """
            exception handling for file operations
            """
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
        usage()