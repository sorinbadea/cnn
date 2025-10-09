from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.layers import MaxPooling2D

class ConvolutionNN:
    def __init__(self, image_path, verbose=False):
        self._image = Image.open(image_path)
        self._verbose = verbose

    def kernel_load(self, array):
        """
        load a specific filter or kernel
        @array: kernel pattern
        """
        self._kernel = np.array(array)
        self._kernel_rows, self._kernel_cols = self._kernel.shape

    def image_resize(self, width=24):
        """
        resize input image to required width
        @param width: width to resize
        """
        ratio = self._image.width / width
        self._image.thumbnail((width, int(self._image.height/ratio)), Image.Resampling.LANCZOS)

    def grayscale(self):
        grayscale = ImageOps.invert(self._image.convert('L'))
        if self._verbose:
            self._image.show()
        self._array = np.array(grayscale)
        self._image_rows, self._image_cols = self._array.shape
        if self._verbose:
            print("initial image matrix")
            print("------ height ", self._array.shape[0], " width ", self._array.shape[1])
            print(self._array)

    def run_convolution(self, normalized_array):
        """
        generator to apply the convolution algorithm
        @param : normalized_array: input image normalized to [0, 1]
        """
        for i in range(self._image_rows - self._kernel_rows + 1):
            for j in range(self._image_cols - self._kernel_cols + 1):
                # Element-wise multiplication and sum
                yield i, j, np.sum(normalized_array[i:i+self._kernel_rows, j:j+self._kernel_cols] * self._kernel)

    def convolution2d(self):
        """
        apply the convolution algorith to the image
        """
        feature_map = np.zeros((self._image_rows - self._kernel_rows + 1, self._image_cols - self._kernel_cols + 1))
        # Normalize image to [0, 1]
        normalized_array = np.round(self._array/255, 2)
        for i, j, conv in self.run_convolution(normalized_array):
                feature_map[i , j] = np.round(conv, 2)
        return feature_map

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
        pooled_map = pooled_map.numpy().squeeze()
        return pooled_map
        
    def process(self, pool_size, pool_stride):
        """
        apply the following on an image:
        normalization, convolution, ReLU, and max_pooling
        @pool_size: the size, (width and height) of the pooling array
        @pool_stride: value to shift on the right and down on each step of max pooling
        """
        self._feature_map = self.convolution2d()
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
            h_pool, w_pool = self._pooled_map.shape
        return self._pooled_map
