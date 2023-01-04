import numpy as np
from numpy import array
from scipy.linalg import svd
mat1=np.array([[[10,8,3],[8,3,9],[5,8,3]]])
mat2=np.array([[[5,8,3],[8,2,9],[5,8,4]]])
print("Addition")
print(np.add(mat1,mat2))
print("Difference")
print(np.subtract(mat1,mat2))
print("Divide")
print(np.divide(mat1,mat2))
print("Multiplication")
print(np.multiply(mat1,mat2))
Array=array([[1,3,5],[2,3,6],[7,2,8]])
A,B,C=svd(Array)
print("Decomposition of Matrix\n",A)
print("Inverse of Matrix\n",B)
print("Transpose of Matrix\n",C)
