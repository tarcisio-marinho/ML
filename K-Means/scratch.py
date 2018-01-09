import numpy as np
import pandas as pd
import matplotlib.ploty as plt
from matplotlib import style

style.use("ggplot")

x = np.array([[1, 2],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(x[:,0], x[:,-1], s=150)
plt.show()