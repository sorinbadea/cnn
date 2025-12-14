from PIL import Image, ImageOps
import numpy as np
from scipy.signal import convolve2d
from tensorflow.keras.layers import MaxPooling2D
import filters

class ConvolutionNN:
    def __init__(self, image_path, verbose=False):
        self._image_path = image_path
        self._image = None
        self._array = None
        self._verbose = verbose
        self._pooled_map = None
        self._kernel = None
        self._activated_map = None

    def kernel_load(self, array):
        """
        load a specific filter or kernel
        @array: kernel pattern
        """
        self._kernel = np.array(array)

    def pre_processing(self, width=128):
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
        image_rows, image_cols = self._array.shape
        if self._verbose:
            print("initial image matrix")
            print("------ height ", image_rows, " width ", image_cols)
            image.show()
            print(self._array)

    def print_array(self, text, array):
        """
        display an array
        """
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
        act_map = self._activated_map.reshape(1, h, w, 1)
        max_pool = MaxPooling2D(pool_size=(pool_size, pool_size), strides=pool_stride, padding='same')
        pooled_map = max_pool(act_map)
        return pooled_map.numpy().squeeze()

    def process(self, pool_size, pool_stride):
        """
        apply the following on an image:
        normalization, convolution, ReLU, and max_pooling
        param @pool_size: the size, (width and height) of the pooling array
        param @pool_stride: value to shift on the right and down on each step of max pooling
        """
        normalized_array = self._array/255.0
        self.print_array("Normalized image matrix", normalized_array)
        feature_map = convolve2d(normalized_array, self._kernel, mode='valid')
        self.print_array("Feature map", feature_map)
    
        # apply RE LU activation function
        self._activated_map = np.maximum(0, feature_map)
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
        self._engine = None
        self._shape_pooled_maps = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._engine
        del self._shape_pooled_maps

    def pre_processing(self):
        """
        Pre-process the image: resize, grayscale, invert if needed
        """
        self._engine = ConvolutionNN(self._image_path, self._verbose)
        self._engine.pre_processing(self._reduce_width)

    def process(self, shape):
        """
        Process the image loaded from image_path;
        returns a map of pooled outputs for each kernel shape
        @param shape : a shape dictionary from filters.py
        """
        self._shape_pooled_maps = {}
        kernel_hash = shape['filters']
        for key in kernel_hash:
            self._engine.kernel_load(kernel_hash[key])
            # run the convolution algorithm per kernel
            self._shape_pooled_maps[key] = self._engine.process(filters.pool_size, filters.stride)
        return self._shape_pooled_maps

