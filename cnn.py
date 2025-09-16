from PIL import Image
import numpy as np

class ConvolutionNN:
    def __init__(self, verbose=False):
        self._verbose = verbose

    def kernel_load(self, array):
        self._kernel = np.array(array)
        self._kernel_rows = self._kernel.shape[0]
        self._kernel_cols = self._kernel.shape[1]

    def image_set(self, array):
        self._array = np.array(array)
        self._image_rows = self._array.shape[0]
        self._image_cols = self._array.shape[1]

    def image_load(self, image_path):
        self._image = Image.open(image_path)
        grayscale = self._image.convert('L')
        self._array = np.array(grayscale)
        self._image_rows = self._array.shape[0]
        self._image_cols = self._array.shape[1]

    def convolution2d(self):
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

    def max_pooling(self, size, stride=3):
        rows, cols = self._activated_map.shape
        pooled_rows = (rows - size) // stride + 1
        pooled_cols = (cols - size) // stride + 1
        pooled_map = np.zeros((pooled_rows, pooled_cols))
        for i in range(pooled_rows):
            for j in range(pooled_cols):
                pooled_map[i, j] = np.max(self._activated_map[i*stride:i*stride+size, j*stride:j*stride+size])
        return pooled_map

    def print_array(self, text, array):
        if self._verbose:
            print("------" + text + "--------")
            print(array)

    def process(self):
        self.convolution2d()
        self.print_array("Feature map", self._feature_map)
        self.ReLU()
        self.print_array("Activated map", self._activated_map)
        self._pooled_map = self.max_pooling(3)
        self.print_array("Pooled map", self._pooled_map)
        return self._pooled_map

    def report(self):
        # Or keep RGB channels (3D array with values 0-255)
        #rgb_array = np.array(image)
        print(f"Image shape: {self._array.shape}")  # (height, width) or (height, width, channels)
        print("--------------------------")
        print(f"Kernel shape: {self._kernel.shape}")
        print("----------------------------")
        print(f"Featured map: {self._feature_map.shape}")
        print(f"Activated map: {self._activated_map.shape}")
        print(f"Pooled map: {self._pooled_map.shape}")
        print(f"pooled map: {self._pooled_map}")
