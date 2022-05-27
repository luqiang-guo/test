import numpy as np


a = np.arange(8)
print(a)
b = np.concatenate(a, axis=-1)
print(b)