from PIL import Image, ImageOps
import numpy as np
from scipy.signal import convolve2d
from tensorflow.keras.layers import MaxPooling2D
import filters
import sys

class ConvolutionNN:
    def __init__(self, image_path, verbose=False):
        self._image_path = image_path
        self._verbose = verbose

    def kernel_load(self, array):
        """
        load a specific filter or kernel
        @array: kernel pattern
        """
        self._kernel = np.array(array)
        self._kernel_rows, self._kernel_cols = self._kernel.shape

    def pre_processing(self, width = 48):
        """
        resize image, convert to gray-scale
        @param width: image width to resize
        """
        self._image = Image.open(self._image_path)
        ratio = self._image.width / width
        self._image.thumbnail((width, round(self._image.height/ratio)), Image.Resampling.LANCZOS)
        """
        convert to grayscale, 255 levels
        0 = black, 255 = white
        """
        image = self._image.convert('L')
        image = ImageOps.invert(image)
        self._array = np.array(image)
        self._image_rows, self._image_cols = self._array.shape
        if self._verbose:
            print("initial image matrix")
            print("------ height ", self._image_rows, " width ", self._image_cols)
            image.show()
            print(self._array)

    def print_array(self, text, array):
        if self._verbose:
            print("------" + text + "------- height ", array.shape[0], " width ", array.shape[1])
            print(array)

    def max_pooling2d(self, pool_size, pool_stride):
        """
        apply max pooling to the activated map using the keras layer
        @pool_size: the size, (width and height) of the pooling array
        @pool_stride: value to shift on the right and down on each step of max pooling
        """
        h, w = self._activated_map.shape
        self._activated_map = self._activated_map.reshape(1, h, w, 1)
        max_pool = MaxPooling2D(pool_size=(pool_size, pool_size), strides=pool_stride, padding='same')
        pooled_map = max_pool(self._activated_map)
        return pooled_map.numpy().squeeze()

    def process(self, pool_size, pool_stride):
        """
        apply the following on an image:
        normalization, convolution, ReLU, and max_pooling
        param @pool_size: the size, (width and height) of the pooling array
        param @pool_stride: value to shift on the right and down on each step of max pooling
        """
        normalized_array = np.round(self._array/255, 2)
        self._feature_map = convolve2d(normalized_array, self._kernel, mode='valid')
        self.print_array("Feature map", self._feature_map)
    
        # apply RE LU activation function
        self._activated_map = np.maximum(0, self._feature_map)
        self.print_array("Activated map", self._activated_map)

        self._pooled_map = self.max_pooling2d(pool_size, pool_stride)
        self.print_array("Pooled map", self._pooled_map)

        h_pool, w_pool = self._pooled_map.shape
        while w_pool > 5:
            self._activated_map = self._pooled_map
            """
            re-apply max pooling
            """
            self._pooled_map = self.max_pooling2d(pool_size, pool_stride)
            self.print_array("Pooled map", self._pooled_map)
            h_pool, w_pool = self._pooled_map.shape

        return self._pooled_map

"""
Wrapper class over ConvolutionNN
"""
class ImageProcessor:
    def __init__(self, image_path, width, verbose=False):
        self._image_path = image_path
        self._verbose = verbose
        self._reduce_width = width

    def pre_processing(self):
        """
        Pre-process the image: resize, grayscale, invert if needed"""
        try:
            self._engine = ConvolutionNN(self._image_path, self._verbose)
            self._engine.pre_processing(self._reduce_width)
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

