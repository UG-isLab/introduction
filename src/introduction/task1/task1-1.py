import numpy as np

a1 = np.array([[1, 2, 3],[4, 5, 6]])
a2 = np.array([[10, 20, 30], [40, 50, 60]])

print(f"result1 : \n {a1+a2}")
print(f"resutl2 : \n {a2-a1}")
print(f"resutl3 : \n {a2*a1}")
print(f"resutl4 : \n {a2/a1}")

a3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result5 = np.dot(a1, a3)
print(f"result5: \n {result5}")
result6 = np.divide(a2, a1)
print(f"result6 : \n {result6}")

