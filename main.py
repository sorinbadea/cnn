import cnn
import sys
import os
from PIL import Image
import filters
import database as data
import analyzer as ana

REDUCED_WIDTH = 48

def usage():
    print("Usage python main.py -f [image file], debug image processing")
    print("      python main.py -d [folder_with images], train the model")
    print("      python main.py -a [image file], analyse mode")

def get_files_from_directory(directory):
    """
    @param: directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files_only:
        yield f

def process(image_path, verbosity):
    """
    Process the image loaded from image_path;
    returns a map of pooled outputs for each kernel shape
    @param image_path : image location
    @param verbosity: enable verbosity
    """
    try:
        conv = cnn.ConvolutionNN(image_path, verbosity)
        conv.image_resize(REDUCED_WIDTH)
        conv.grayscale()
        pooled_maps = {}
        kernel_hash = filters.kernels_digit_one['filters']
        for (key, i) in zip (kernel_hash, range(len(kernel_hash))):
            conv.kernel_load(kernel_hash[key])
            # run the convolution algorithm per kernel
            pooled = conv.process(filters.kernels_digit_one['pool_size'],
                                filters.kernels_digit_one['stride'])
            if verbosity:
                print("pooled output for kernel:", i)
                print(pooled)
            pooled_maps[key] = pooled
        return pooled_maps
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
    print("processing image stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-f":
            """
            single image processing mode
            """
            image_path = sys.argv[2]
            pooled_map = process(image_path, True)
            for key in pooled_map:
                print("Pooled output for kernel:", key)
                for row in pooled_map[key]:
                    print(row)
            print("Processing completed.")

        elif sys.argv[1] == "-a":
            image_path = sys.argv[2]
            pooled_map = process(image_path, False)

            """
            call the analyzer module
            """
            results = ana.evaluate(pooled_map, True)
            ana.evaluate_results(results)

        elif sys.argv[1] == "-d":
            image_path = sys.argv[2]
            """
            training mode, update a table with pooled data for each kernel shape
            """
            db = data.DataBaseInterface('localhost','myapp','postgres','password',5432)
            for key in filters.kernels_digit_one['filters']:
                if db.create_table(key) is False:
                    print(f"❌ Error creating table '{key}', exit training process")
                    sys.exit(1)
            """
            start processing all images in the specified folder
            """
            for image in get_files_from_directory(image_path):
                print("Processing image:", image)
                polled_map = process(image_path + "/" + image, False)
                for key in polled_map:
                    for row in polled_map[key]:
                        # convert from numpy array to list
                        db.insert_data(key, row.tolist())
            db.database_disconnect()
        else:
            usage()
    else:
        usage()