from numpy.linalg import svd
from numpy import mat
if __name__=="__main__":
    X = mat([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    U, S, T = svd(X)
    print(U[:,0:3].round(1))
    print(T[0:3,:].round(1))