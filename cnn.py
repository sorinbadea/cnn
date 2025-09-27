from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

class ConvolutionNN:
    def __init__(self, image_path, verbose=False):
        self._image = Image.open(image_path)
        self._verbose = verbose

    def kernel_load(self, array):
        """
        load a speciffic filter or kernel
        @array: kernel pattern
        """
        self._kernel = np.array(array)
        self._kernel_rows, self._kernel_cols = self._kernel.shape

    def image_resize(self, width=24):
        """
        resize input image to width and invert the gray colors
        @param width: width to resize
        """
        ratio = self._image.width / width
        self._image.thumbnail((width, int(self._image.height/ratio)), Image.Resampling.LANCZOS)
        grayscale = ImageOps.invert(self._image.convert('L'))
        if self._verbose:
            self._image.show()
        self._array = np.array(grayscale)
        self._image_rows, self._image_cols = self._array.shape
        if self._verbose:
            print("initial image matrix")
            print("------ height ", self._array.shape[0], " width ", self._array.shape[1])
            print(self._array)

    def convolution2d(self):
        """
        apply the convolution algorith to the image
        """
        # Normalize image to [0, 1]
        normalized_array = np.round(self._array / 255, 2)
        feature_map = np.zeros((self._image_rows - self._kernel_rows + 1, self._image_cols - self._kernel_cols + 1))
        for i in range(self._image_rows - self._kernel_rows + 1):
            for j in range(self._image_cols - self._kernel_cols + 1):
                # Element-wise multiplication and sum
                conv_value = np.sum(normalized_array[i:i+self._kernel_rows, j:j+self._kernel_cols] * self._kernel)
                feature_map[i , j] = conv_value
        self._feature_map = feature_map

    def ReLU(self):
        self._activated_map = np.maximum(0, self._feature_map)

    def max_pooling(self, size=3, stride=2):
        """
        apply the max pooling step, retrive the maximum value from
        the featured map (convolution result)
        """
        rows, cols = self._activated_map.shape
        pooled_rows = (rows - size) // stride + 1
        pooled_cols = (cols - size) // stride + 1
        pooled_map = np.zeros((pooled_rows, pooled_cols))
        for i in range(pooled_rows):
            for j in range(pooled_cols):
                pooled_map[i, j] = np.max(self._activated_map
                    [i * stride: i * stride + size, j * stride : j * stride + size])
        return pooled_map

    def print_array(self, text, array):
        if self._verbose:
            print("------" + text + "------- height ", array.shape[0], " width ", array.shape[1])
            print(array)

    def process(self, pool_size, pool_stride):
        """
        apply the process of image
        normalization, convolution, ReLU, and max_pooling
        @pool_size: the size, width and height of the pooling array
        @pool_stride: value to shift on the right and down on each step of max pooling
        """
        self.convolution2d()
        self.print_array("Feature map", self._feature_map)
        self.ReLU()
        self.print_array("Activated map", self._activated_map)
        self._pooled_map = self.max_pooling(pool_size, pool_stride)
        h_pool, w_pool = self._pooled_map.shape
        # reduce pooled map to max 4 elements
        self.print_array("Pooled map", self._pooled_map)
        while h_pool * w_pool > 4:
            self._activated_map = self._pooled_map
            self._pooled_map = self.max_pooling(pool_size, pool_stride)
            self.print_array("Pooled map", self._pooled_map)
            h_pool, w_pool = self._pooled_map.shape
        #prediction = tf.reduce_mean(self._pooled_map).numpy()
        #print("Prediction score (mean pooled activation):", prediction)
        return self._pooled_map

