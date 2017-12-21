# U*W + b > 0 || < 0 || == 0


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
        
df = pd.read_csv('data/breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) # remove ID

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

