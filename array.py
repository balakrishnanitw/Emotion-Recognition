from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
a = read('sample.wav')
b = np.array(a[1], dtype=float)
plt.figure()
plt.plot(b)
plt.show()
