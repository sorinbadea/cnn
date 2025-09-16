import numpy as np
import cnn
import filters
import shapes

# "1".shape checking functions
def find_vertical_identical_positions(array):
    hits = 0
    rows, cols = array.shape
    for j in range(cols):
        for i in range(rows - 1):
            if array[i, j] != 0:
                if array[i, j] == array[i + 1, j]:
                    hits += 1
    return hits

def find_diagonal_identical_positions(array):
    hits = 0
    rows, cols = array.shape
    for j in range(cols - 1):
        for i in range(rows - 1):
            if array[i, j] != 0:
                if array[i, j] == array[i + 1, j + 1]:
                    hits += 1
    return hits

def main():
    kernels = [filters.kernel_one_1, filters.kernel_one_2]
    flat_layer = []
    conv = cnn.ConvolutionNN(True)
    conv.image_set(shapes.image2)
    for kernel in kernels:
        conv.kernel_load(kernel)
        pooled = conv.process()
        flat_layer.append(pooled)
    ("----------- Evaluating pooled matrixes out, 3 max ----------")
    for array in flat_layer:
        print("vertical hits ", find_vertical_identical_positions(array))
        print("diagonal hits ", find_diagonal_identical_positions(array))

if __name__ == "__main__":
    main()
