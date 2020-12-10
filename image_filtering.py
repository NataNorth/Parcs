import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve
from Pyro4 import expose


edge_detection_operator = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]))

edge_detection_operator2 = np.array(([1, 1, 1],[1, 1, 1],[1, 1, 1]))

small_test = np.array(([1, 2, 3],[4, 5, 6],[7, 8, 9]))

def pad_image(image_matrix):
    result = np.zeros((image_matrix.shape[0] + 2, image_matrix.shape[1] + 2))
    result[1:image_matrix.shape[0]+1, 1:image_matrix.shape[1]+1] = image_matrix
    return  result.tolist()

def compute_pixel(cell, kernel):
    # if cell.shape != kernel.shape:
    #     raise ValueError("Wrong shape")
    acc = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            acc += cell[i][j]*kernel[-i-1][-j-1]
    return acc

def split_into_rows(matrix):
    rows = []
    columns = len(matrix[0])
    for i in range(1, len(matrix) - 1):
        rows.append((np.array(matrix)[i-1:i+2, :]).tolist())
    return rows

def split_row_into_cells(row):
    cells = []
    for i in range(1, len(row[0]) - 1):
        cells.append((np.array(row)[0:3,i-1:i+2]).tolist())
    return cells

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.rows = 0
        self.columns = 0
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        matrix = self.read_input()
        matrix = pad_image(matrix)
        rows = split_into_rows(matrix)
        computed_rows = []

        for i in range(len(rows)):
            computed_rows.append(self.workers[i%len(self.workers)].computeRow(rows[i]))

        self.write_output(computed_rows)

        print("Job Finished")

    @staticmethod
    @expose
    def computeRow(row):
        cells = split_row_into_cells(row)
        computed = []
        for cell in cells:
            computed.append(compute_pixel(cell, edge_detection_operator))
        return computed

    def read_input(self):
        input = np.loadtxt(self.input_file_name, dtype='f', delimiter=' ')
        self.rows = len(input)
        self.columns = len(input[0])
        return input

    def write_output(self, output):
        f = open(self.output_file_name, 'w')
        for row in output:
            for pixel in row.value:
                f.write(str(int(pixel)))
                f.write(' ')
            f.write('\n')
        f.close()
        # mat = np.matrix(np.array(output))
        # np.savetxt(self.output_file_name, mat, fmt='%d')

if __name__ == '__main__':
    # test_matrix = np.loadtxt("image.txt")
    # test_matrix = pad_image(test_matrix)
    # rows = split_into_rows(test_matrix)
    # cells = [split_row_into_cells(rows[i]) for i in range(len(rows))]
    # matrix = [[0 for i in range(len(test_matrix[0])-2)] for i in range(len(test_matrix) - 2)]
    # for i in range(len(cells)):
    #     for j in range(len(cells[0])):
    #         matrix[i][j] = compute_pixel(cells[i][j], edge_detection_operator)
    # np.savetxt("output.txt", convolve(test_matrix, edge_detection_operator), delimiter=' ', fmt="%d")
    # np.savetxt("output2.txt", np.array(matrix), delimiter=' ', fmt="%d")
    worker = Solver()
    solver = Solver(workers=[worker], input_file_name='image.txt', output_file_name='output2.txt')
    solver.solve()
