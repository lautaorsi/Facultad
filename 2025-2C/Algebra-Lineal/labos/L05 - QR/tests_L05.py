### Agregados

def norma(x,p):
    res = 0
    if p == np.inf:
        res = abs(x[0])
        for i in range(len(x)):
            if(res < abs(x[i])):
                res = abs(x[i])
        return res
    else:
        for i in range(len(x)):
            res += np.power(abs(x[i]),p)
        return np.power(res, 1/p)


def traspuesta(matriz):
    nueva_matriz = np.zeros(matriz.shape)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[0]):
            nueva_matriz[j][i] = matriz[i][j]
    return nueva_matriz

def multiplicacionVectorial(x,y):
    assert x.shape == y.shape
    res = 0
    for i in range(x.shape[0]):
        res += x[i]*y[i]
    return res

def tolerar(x, tol):
    if x.shape == ():          #Si x es un escalar
        if x <= tol:
            return 0
        return x

    for i in range(x.shape[0]): #Si x es un vector
        if x[i] <= tol:
            x[i] = 0
    return x
    
def subVector(v, k, m):         #Dado un vector de Nx1 y dos indices retorna el vector desde k hasta m (incluidos ambos)
    subvector = np.zeros((1,m-k+1))
    for i in range(k, m+1):
        subvector[0][i-k] = v[i]
    return subvector

def sign(x):
    if (x < 0):
        return -1
    if (x > 0):
        return 1
    return 0


### Funciones L05-QR
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """

    if(A.shape[0] != A.shape[1]):
        return None

    #Me parece mas facil trabajar sobre filas en vez de columnas,
    #entonces podemos hacer toda la descomp sobre At (A traspuesta) en vez de A
    #pero OJO, porque Q lo voy a escribir en filas tambien, entonces tenemos que trasponerlo nuevamente
    A = traspuesta(A)
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    nops = 0
    
    R[0][0] = tolerar(norma(A[0],2),tol)

    Q[0] = A[0] / R[0][0]
    

    for j in range(1, A.shape[0]):    
        temp = A[j].copy()
        for k in range(j):
            R[k][j] = tolerar(multiplicacionVectorial(Q[k],temp),tol)
            temp -= R[k][j] * Q[k]
        
        R[j][j] = tolerar(norma(temp, 2),tol)
        Q[j] = temp / R[j][j]    

    Q = traspuesta(Q)
    
    if(retorna_nops):
        return Q,R,nops
    else:
        return Q,R

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """

    m = A.shape[0]
    n = A.shape[1] 

    if(m < n):
        return None
    
    R = A
    Q = np.eye(m,m)

    for k in range(n):
        x = subVector(R[:,k], k, m-1)           #x es un vector columna
        
        alfa = -sign(x[0][0]) * norma(x, 2)

        u = x - alfa * (np.eye(0,m-k))
        
        if(norma(u, 2) > tol):
            u = u / norma(u, 2)
            
            H[:,k] = np.eye(m-k+1, m-k+1) - 2 * multiplicacionVectorial(u,u)
            
            extH = armarExtI(H,k)

            R = multiplicacionMatricial(extH,R)

            Q = multiplicacionMatricial(Q,traspuesta(extH))
    
    return Q,R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

    if(metodo == 'RH'):
        return QR_con_HH(A,tol)
    if(metodo == 'GS'):
        return QR_con_GS(A,tol)
    
    return None


# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# # --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)