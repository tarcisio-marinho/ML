# Y = M*X + B

# M = média(x) * média(y) - média(x*y) / média(x) **2 - média(x**2)

# B = média(y) - M * média(x)

# Grafico
#plt.scatter(xs, ys)
#plt.show()


from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

def best_fit_slop_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - (mean(xs*ys))) /
          ((mean(xs)**2) - mean(xs**2)) )
    
    b = mean(ys) - m*mean(xs)

    return m, b

if __name__ == "__main__":
        
    # np array
    xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
    ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

    m, b = best_fit_slop_and_intercept(xs, ys)
    regression_line = [(m*x) + b for x in xs]

    predict_x = 8

    predict_y = (m*predict_x) + b


    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y)
    plt.plot(xs, regression_line)
    plt.show()