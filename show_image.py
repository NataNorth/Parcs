from PIL import Image, ImageOps

import numpy as np

if __name__ == '__main__':
    mat = np.genfromtxt('output.txt', delimiter=' ')[:,:-1]
    Image.fromarray(mat).show()
