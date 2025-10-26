import cnn
import sys
import os
from PIL import Image
import filters
import database as data
import analyzer as ana

REDUCED_WIDTH = 128
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

def get_shape_name():
    """
    returns the index of the kernel set to use for training
    based on the element name provided as command line argument
    e.g. "digit '1'"
    """
    if len(sys.argv) < 4:
        usage()
    element_to_match = sys.argv[3]
    for i, shape in enumerate(filters.shapes):
        if shape['name'] == element_to_match:
            return i
    print(f"Error: element to train '{element_to_match}' not found")
    usage()

class ImageProcessor:
    def __init__(self, image_path, width, verbose=False):
        self._image_path = image_path
        self._verbose = verbose
        self._reduce_width = width

    def pre_processing(self):
        try:
            self._engine = cnn.ConvolutionNN(self._image_path, self._verbose)
            self._engine.pre_processing(REDUCED_WIDTH)
            return self._engine
        except FileNotFoundError:
            print(f"Error: File '{self._image_path}' not found")
            # Handle the error (set default, raise custom exception, etc.)
        except Image.UnidentifiedImageError:
            print(f"Error: '{self._image_path}' is not a valid image file")
        except PermissionError:
            print(f"Error: Permission denied accessing '{self._image_path}'")
        except Exception as e:
            print(f"Unexpected exception: {e}")
        return None

    def process(self, shape_index):
        """
        Process the image loaded from image_path;
        returns a map of pooled outputs for each kernel shape
        @param shape_index : shape number from filters.py
        """
        pooled_maps = {}
        kernel_hash = filters.shapes[shape_index]['filters']
        for key in kernel_hash:
            self._engine.kernel_load(kernel_hash[key])
            # run the convolution algorithm per kernel
            pooled = self._engine.process(filters.pool_size, filters.stride)
            pooled_maps[key] = pooled
        return pooled_maps

def verdict(cosine_result, eucl_result):
    """
    issue the final verdict
    for euclidian distance, get the maximum match from all tried filters
    for cosine similarity get the shape with the maximum similarities
    @param cosine_result - name of the shape with the best cosine similarity
    @param eucl_result; hash of euclidian matches/shape
    """
    cosine_match = max(cosine_result, key=cosine_result.get)
    eucl_dist_match = max(eucl_result, key=eucl_result.get)
    eucl_percent = round(eucl_result[eucl_dist_match] * 100)

    if eucl_percent > 66 and cosine_match == eucl_dist_match:
        # ideal case, both evaluation methods matches
        # evaluate other shape confidence
        print(cosine_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    elif eucl_percent > 70 and cosine_match != eucl_dist_match:
        # evaluate other shape confidence
        for key in eucl_result:
            if cosine_match == key and (eucl_result[key]*100) >= 50:
                 #consider cosine match if euclidian is still important
                 print(cosine_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)
                 return
        # higher euclidian match but not matching the cosine, take eculidian
        print(eucl_dist_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    elif eucl_percent > 30 and eucl_percent < 71 and cosine_match == eucl_dist_match:
        # low euclidian confidence, take cosine
        print(eucl_dist_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    else:
        # unknown pattern
        print("unknow patern, low euclidian confidence", eucl_percent, "% cosine evaluation", cosine_match)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == "-f":
            """
            single image processing mode
            """
            image_path = sys.argv[2]
            img_processor = ImageProcessor(image_path, REDUCED_WIDTH, True)
            if img_processor.pre_processing() == None:
                sys.exit(1)
            for shape_index in range(len(filters.shapes)):
                print("shape index ", shape_index + 1)
                pooled_map = img_processor.process(shape_index)
                # display results
                for key in pooled_map:
                   print("Pooled output for kernel:", key)
                   for row in pooled_map[key]:
                        print(row)
            print("Processing completed.")

        elif sys.argv[1] == "-a":
            image_path = sys.argv[2]
            eucl_result = {}
            cosine_result = {}
            img_processor = ImageProcessor(image_path, REDUCED_WIDTH, False)
            if img_processor.pre_processing() == None:
                sys.exit(1)
            for shape_index in range(len(filters.shapes)):
                pooled_map = img_processor.process(shape_index)
                """
                call the analyzer module
                """
                euclidian_result, similarity = ana.evaluate(pooled_map, shape_index, False)
                eucl_result[filters.shapes[shape_index]['name']] = euclidian_result
                cosine_result[filters.shapes[shape_index]['name']] = similarity
                print(f"Euclidian dist confidence {round(euclidian_result * 100)} % for {filters.shapes[shape_index]['name']}")
            # issue final verdict
            verdict(cosine_result, eucl_result)

        elif sys.argv[1] == "-d":
            image_path = sys.argv[2]
            """
            training mode, update a table with pooled data for each kernel shape
            """
            shape_index = get_shape_name()
            db = data.DataBaseInterface('localhost','myapp','postgres','password',5432)
            for key in filters.shapes[shape_index]['filters']:
                if db.create_table(key) is False:
                    print(f"❌ Error creating table '{key}', exit training process")
                    sys.exit(1)
            """
            start processing all images in the specified folder
            """
            for image in get_files_from_directory(image_path):
                img_processor = ImageProcessor(image_path + "/" + image, REDUCED_WIDTH, False)
                if img_processor.pre_processing() == None:
                    continue
                print("Processing image:", image)
                polled_map = img_processor.process(shape_index)
                for key in polled_map:
                    for row in polled_map[key]:
                        # convert from numpy array to list
                        db.insert_data(key, row.tolist())
            db.database_disconnect()
        else:
            usage()
    else:
        usage()