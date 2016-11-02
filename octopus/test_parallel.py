import numpy as np
from multiprocessing import Pool


def f(i):
    x=10
    a = np.zeros(x)
    for i in range(x):
        a[i] = i
    return a

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [0, 1, 2]))
