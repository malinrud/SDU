import numpy as np

"""""
a = np.full((2, 3), 4)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.eye(2, 3)
d = a + b + c

print(d)
"""

a = np.array([[1,2,3,4,5],
              [5,4,3,2,1],
              [6,7,8,9,0],
              [0,9,8,7,6]])

print(np.sum(a))
print(np.transpose(a))

