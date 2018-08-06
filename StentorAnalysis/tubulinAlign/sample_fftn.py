import numpy as np
import matplotlib.pyplot as plt

[X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
                     2 * np.pi * np.arange(200) / 34)
S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
FS = np.fft.fftn(S)
plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
plt.show()
