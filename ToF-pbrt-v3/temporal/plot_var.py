import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    origin = [0.400814, 1.191135, 0.800316, 6.029873, 2.367448, 25.984900]
    darts = [0.003807, 0.065746, 0.049246, 0.035502, 0.044579, 0.080963]
    ts = np.linspace(0.1, 0.6, 6)
    plt.scatter(ts, origin, color = 'red', s = 5)
    plt.scatter(ts, origin, color = 'blue', s = 5)
    plt.plot(ts, origin, color = 'red', label = 'variance for original method')
    plt.plot(ts, origin, color = 'blue', label = 'variance for DARTS')
    plt.legend()
    plt.grid(axis = 'both')
    plt.xlabel('scattering coefficient')
    plt.ylabel('variance')
    plt.show()