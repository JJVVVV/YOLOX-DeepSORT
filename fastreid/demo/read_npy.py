import numpy as np
import os
root = 'demo_output\\'
files = [root+name for name in os.listdir(root)]
for file in files:
    a = np.load(file)
    print(a.shape)
    print(a[:, :10])
