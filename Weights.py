import numpy as np

# Xavier Normalized Initialization
def xavier_norm(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))
