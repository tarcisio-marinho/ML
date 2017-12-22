# U*W + b > 0 || < 0 || == 0 # decide se o novo ponto
#  está abaixo ou acima da linha, determinando o conjunto a que o novo
# ponto pertence

# A = [1, 3]
# B = [4, 2]
# dot product of two vectors: A*B
# (1*4) + (3*2) = 10 -> valor escalar

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
        
df = pd.read_csv('data/breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) # remove ID

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

