import numpy as np

a = np.array([[1, 3, 7, 4], [1, 2, 1, 9], [3, 1, 5, 6]])
a=sorted(a,key=lambda x: (x[0], x[1]))
print(a)

