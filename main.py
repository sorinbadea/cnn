import sys
import os
from pathlib import Path
import filters
import cnn
import database as data
import analyzer as ana
import verdict as vd
from database import DataBaseInterface

REDUCED_WIDTH = 128

def usage() -> None:
    print("Usage python main.py -d [image file], debug image processing steps")
    print("      python main.py -t [folder with images] [shape to train] train the model")
    print("      python main.py -a [image file | image_path], analyse mode, run all the filters and output a confidence percent")
    sys.exit(1)

def get_files_from_directory(directory):
    """
    @param: directory: Directory path
    Generator to retrieve a list of files in the specified directory.
    """
    files_only = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    yield from files_only

def get_shape_dict(shape2match) -> int:
    """
    returns the shape dictionary for training
    based on the element name provided as command line argument
    e.g. "digit '1'"
    """
    for i, shape in enumerate(filters.shapes):
        if shape['name'] == shape2match:
            return filters.shapes[i]
    print(f"Error: element to train '{shape2match}' not found")
    return None
    
def process_and_analyse_image(image_path, db_if, verbose=False) -> None:
    """
    process and analyse a single image
    @param image_path: path to the image file
    @param db: database interface
    @param verbose: verbose mode
    """
    eucl_result = {}
    cosine_result = {}
    with cnn.ImageProcessor(image_path, REDUCED_WIDTH, False) as img_proc:
        try:
            img_proc.pre_processing()
            for shape in filters.shapes:
                shape_pooled_map = img_proc.process(shape)
                """
                call the analyzer module
                """
                result = ana.evaluate(shape_pooled_map, shape, db_if, verbose)
                """
                fill all the results for verdict processing
                """
                eucl_result[shape['name']] = result['euclidian']
                cosine_result[shape['name']] = result['cosine']
                print(
                    f"Euclidian evaluation confidence {round(result['euclidian'] * 100, 2)} % for {shape['name']}"
                    )
        except Exception as e:
            print(f"Unexpected exception during processing image '{image_path}': {e}")
            return

    # issue verdict
    print("=====> Results")
    print(f"--> Cosine evaluation '{max(cosine_result, key=cosine_result.get)}'")
    res = vd.verdict(cosine_result, eucl_result)
    print(f"'{res}' image in file '{image_path}'")

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        usage()
    # parse command line arguments
    if sys.argv[1] == "-d":
        """
        single image processing mode
        """
        image_path = sys.argv[2]
        with cnn.ImageProcessor(image_path, REDUCED_WIDTH, True) as img_processor:
            try:
                img_processor.pre_processing()
                for shape in filters.shapes:
                    print("shape index ", shape['name'])
                    shape_pooled_map = img_processor.process(shape)
                    # display results
                    for key, values in shape_pooled_map.items():
                        print("Pooled output for kernel:", key)
                        for row in values:
                            print(row)
                print("Processing completed.")
            except Exception as e:
                print(f"Unexpected exception during processing image '{image_path}': {e}")

    elif sys.argv[1] == "-a":
        image_path = sys.argv[2]
        with DataBaseInterface('localhost','myapp','postgres','password',5432) as db:
            db.load_trained_data()
            """
            check if image_path is a file or a directory
            """
            if Path(image_path).is_file():
                process_and_analyse_image(image_path, db, True)
            elif Path(image_path).is_dir():
                for image in get_files_from_directory(image_path):
                    process_and_analyse_image(image_path + "/" + image, db, False)
                    print()
            else:
                print(f"Error: '{image_path}' is neither a valid file nor a directory")

    elif sys.argv[1] == "-t":
        if len(sys.argv) < 4:
            usage()
        image_path = sys.argv[2]
        if not Path(image_path).is_dir():
            print(f"Error: '{image_path}' is not a valid directory")
            usage()
        """
        training mode, update a table with pooled data for each kernel shape
        """
        shape = get_shape_dict(sys.argv[3])
        if shape is None:
            usage()
        # database connection
        with data.DataBaseInterface('localhost','myapp','postgres','password',5432) as db:
            # cleanup tables
            for key in shape['filters']:
                db.create_table(key)
            """
            start processing all images in the specified folder
            """
            for image in get_files_from_directory(image_path):
                with cnn.ImageProcessor(image_path + "/" + image, REDUCED_WIDTH, False) as img_processor:
                    try:
                        img_processor.pre_processing()
                        print("Processing image:", image)
                        shape_polled_map = img_processor.process(shape)
                        for key, values in shape_polled_map.items():
                            for row in values:
                                # convert from numpy array to list
                                db.insert_data(key, row.tolist())
                    except Exception as e:
                        print(f"Unexpected exception during processing image '{image}': {e}")
                        continue
    else:
        usage()
