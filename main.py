import cnn
import sys
import os
from PIL import Image
import filters
import database as data
import analyzer as ana

REDUCED_WIDTH = 48

def usage():
    print("Usage python main.py -f [image file], debug image processing steps")
    print("      python main.py -d [folder_with images] [element to train] train the model")
    print("      python main.py -a [image file], analyse mode, run all the filters and output a confidence percent")
    sys.exit(1)

def get_files_from_directory(directory):
    """
    @param: directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files_only:
        yield f

def get_kernel_set_name():
    """
    returns the index of the kernel set to use for training
    based on the element name provided as command line argument
    e.g. "digit '1'"
    """
    if len(sys.argv) < 4:
        usage()
    element_to_match = sys.argv[3]
    for kernel_set in range(len(filters.kernels)):
        if filters.kernels[kernel_set]['name'] == element_to_match:
            print(f"Training mode for element '{element_to_match}'")
            return kernel_set
    print(f"Error: element to train '{element_to_match}' not found")
    usage()

def process(image_path, kernel_set, verbosity):
    """
    Process the image loaded from image_path;
    returns a map of pooled outputs for each kernel shape
    @param image_path : image location
    @kernel_set: kernel or filter to apply
    @param verbosity: enable verbosity
    """
    try:
        conv = cnn.ConvolutionNN(image_path, verbosity)
        conv.image_resize(REDUCED_WIDTH)
        conv.grayscale()
        pooled_maps = {}
        kernel_hash = filters.kernels[kernel_set]['filters']
        for key, i in zip (kernel_hash, range(len(kernel_hash))):
            conv.kernel_load(kernel_hash[key])
            # run the convolution algorithm per kernel
            pooled = conv.process(filters.kernels[kernel_set]['pool_size'],
                                filters.kernels[kernel_set]['stride'])
            if verbosity:
                print("pooled for kernel:", i+1)
                print(pooled)
            pooled_maps[key] = pooled
        return pooled_maps
        """
        exception handling
        """
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found")
        # Handle the error (set default, raise custom exception, etc.)
    except Image.UnidentifiedImageError:
        print(f"Error: '{image_path}' is not a valid image file")
    except PermissionError:
        print(f"Error: Permission denied accessing '{image_path}'")
    except Exception as e:
        print(f"Unexpected exception: {e}")
    print("processing image stopped")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == "-f":
            """
            single image processing mode
            """
            image_path = sys.argv[2]
            for kernel_set in range(len(filters.kernels)):
                print("kernel set ", kernel_set)
                pooled_map = process(image_path, kernel_set, False)
                # display results
                for key in pooled_map:
                   print("Pooled output for kernel:", key)
                   for row in pooled_map[key]:
                        print(row)
            print("Processing completed.")

        elif sys.argv[1] == "-a":
            image_path = sys.argv[2]
            # hash of known patter; (digit1, house) and nb of matches/total kernels
            match_ratio = {}
            for kernel_set in range(len(filters.kernels)):
                pooled_map = process(image_path, kernel_set, False)
                """
                call the analyzer module
                """
                result = ana.evaluate(pooled_map, True)
                match_ratio[filters.kernels[kernel_set]['name']] = result
                print(f"Confidence result {round(result * 100)} % for {filters.kernels[kernel_set]['name']}")
            """
            get the maximum match from all tried filters
            """
            match_max = max(match_ratio, key=match_ratio.get)
            if round(match_ratio[match_max] * 100) < 66:
                print("unknown pattern, confidence result ", round(match_ratio[match_max] * 100), "%")
            else:
                print(match_max, " with a confidence of ", round(match_ratio[match_max] * 100), "%")
            
        elif sys.argv[1] == "-d":
            image_path = sys.argv[2]
            """
            training mode, update a table with pooled data for each kernel shape
            """
            kernel_set = get_kernel_set_name()
            db = data.DataBaseInterface('localhost','myapp','postgres','password',5432)
            for key in filters.kernels[kernel_set]['filters']:
                if db.create_table(key) is False:
                    print(f"❌ Error creating table '{key}', exit training process")
                    sys.exit(1)
            """
            start processing all images in the specified folder
            """
            for image in get_files_from_directory(image_path):
                print("Processing image:", image)
                polled_map = process(image_path + "/" + image, kernel_set, False)
                for key in polled_map:
                    for row in polled_map[key]:
                        # convert from numpy array to list
                        db.insert_data(key, row.tolist())
            db.database_disconnect()
        else:
            usage()
    else:
        usage()