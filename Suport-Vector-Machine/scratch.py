# U*W + b > 0 || < 0 || == 0 # decide se o novo ponto
#  está abaixo ou acima da linha, determinando o conjunto a que o novo
# ponto pertence

# A = [1, 3]
# B = [4, 2]
# dot product of two vectors: A*B
# (1*4) + (3*2) = 10 -> valor escalar

import numpy as np 
import matplotlib.ploty as plt

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if(self.visualization):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    
    # train method
    def fit(self, features):

        # sinal (x*w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)  
        return classification