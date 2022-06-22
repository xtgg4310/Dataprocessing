import numpy as np

a = np.array(1)
c = np.zeros(1)
b = np.array([2])
# a=np.array([a])
# c=np.array([c])
print(a, c)
a = np.append(a, c)
print(a)
