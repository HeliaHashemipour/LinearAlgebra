"""
Solve linear system using LU decomposition.

Author:
    -Helia Sadat Hashemipour
"""
import numpy as np

# Split the first line of input.
# n1 is size of matrix A(coefficient matrix) and rest of the line (m1) is entry of the vectors
n1, m1 = input("").split()
n1 = int(n1)
m1 = int(m1)


# Determining the matrix A ,from the input
def matrix_A():
    matrixA = []
    for i in range(n1):
        row = list(map(int, input().split()))
        matrixA.append(row)
    return matrixA


# Determining the entries of solution set b,from the input
def matrix_b():
    vectorB = []
    for i in range(m1):
        row = list(map(int, input().split()))
        vectorB.append(row)
    return vectorB


a = matrix_A()


# LU Factorization for calculating matrix L & U.
def LU():
    A2 = np.array(a)
    U = np.zeros((n1, n1))
    L = np.eye(n1)
    for j in range(n1):
        U[0][j] = A2[0][j]
        L[j][0] = A2[j][0] / U[0][0]

    for i in range(n1):
        #U
        for j in range(i,n1):
            sum1 = 0.0
            for p in range(i):
                sum1 = sum1 + (L[i][p]*U[p][j])
            U[i][j] = A2[i][j] - sum1
       #L
        for j in range(i, n1):
            sum2 = 0.0
            for p in range(i):
                sum2 = sum2 + (L[j][p] * U[p][i])
            L[j][i] = (A2[j][i] - sum2) / U[i][i]

    return L, U


b = matrix_b()
for i in range(len(b)):
    ba = np.array(b[i])


    # Forward Substitution method for finding solution of Ly=b(rom the top down).
    # L must be a lower triangular matrix
    def fs(L, b):
        # Allocate space with zero
        Y = np.zeros(len(ba))
        for i in range(n1):
            sum = 0.0
            for k in range(i):
                sum = sum + (L[i][k] * Y[k])
            Y[i] = (b[i] - sum) / L[i][i]
        return Y


    # Backward Substitution method for finding solution of Ux=y(from the bottom up).
    # U must be an upper triangular matrix
    def bs(U, y):
        # Allocate space with zero
        X = np.zeros(len(y))
        # Looping over rows from the bottom up.
        for i in range(n1 - 1, -1, -1):
            sum = 0.0
            for k in range(i + 1, n1):
                sum = sum + (U[i][k] * X[k])
            X[i] = (y[i] - sum) / U[i][i]
        return X


    L, U = LU()
    Y = fs(L, ba)
    X = bs(U, Y)
    Org = np.around(X, decimals=2)
    print(*Org)
