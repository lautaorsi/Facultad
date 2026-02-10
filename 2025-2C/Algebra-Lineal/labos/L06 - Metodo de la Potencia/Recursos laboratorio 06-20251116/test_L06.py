# Test L06-metpot2k, Aval

import numpy as np

def calcularAx(matriz,vector):
    res = np.zeros(len(vector))
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            res[i] += (matriz[i][j] * vector[j])
    return res


def norma(x, p):
    if p == 1: #norma 1
       return sum(abs(i) for i in x) 
       
    if p == 'inf': #norma infinito
        acum = 0
        for r in range(x.shape[0]):
            sum_abs=0
            for c in range(x.shape[1]):
                sum_abs += abs(x.shape[1])
            acum = max(acum, sum_abs)
        return acum

    #Norma generica
    return (sum(i**p for i in x ))**(1/p)


def traspuesta(matriz):
    nueva_matriz = np.zeros(A.shape)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            nueva_matriz[j][i] = matriz[i][j]
    return nueva_matriz

def multiplicar_matrices(A, B):

    A = np.array(A)
    B = np.array(B)

    m, n = A.shape
    n2, p = B.shape

    res = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            suma = 0
            for k in range(n):
                suma += A[i, k] * B[k, j]
            res[i, j] = suma

    return res

def multiplicar_vectores(v,w):
    if(v.len() != w.len()):
        return None

    res = 0

    for i in range(v.len()):
        res += v[i] * w[i]
    
    return res




def f(k,A,w):
    for i in range(k):
        w_prima = calcularAx(A,w)

        if(norma(w_prima, 2) > 0):
            w = w_prima/norma(w_prima,2)
        else:
            w = np.zeros((n,1))
    
    return w

def metpot2k(A,tol=1e-15,K=1000):
    #A es nxn
    if(A.shape[0] != A.shape[1]):
        return None
    
    n = A.shape[0]

    #Armamos vector random en R^n
    v = np.random.rand(n,1)
    v_prima = f(2,A,v)
    e = multiplicar_vectores((v_prima), v)
    k = 0
    while(abs(e-1) > tol and k < K):
        v = v_prima
        v_prima = f(2,A,v)
        e = multiplicar_vectores((v_prima), v)
        k += 1
    
    autovalor = multiplicar_matrices(multiplicar_matrices((v_prima), A),v_prima)
    e = e - 1
    return v, autovalor, k, e

def diagRH(A,tol=1e-15,K=1000):
    
    v1, lam1, k, e1 = metpot2k(A,tol,K)

    Hv1 = np.eye(A.shape) - 2 * (((e1-v1) * (e1-v1))/norma(e1-v1)** 2)
    if n == 2:
        S = Hv1
        D = Hv1 * A * traspuesta(Hv1)
    else:
        B = Hv1 * A * traspuesta(Hv1)
        A_prima  = B[2:n][2:n]
        
        S_prima,D_prima = diagRH(A_prima,tol,K)
        
        D = np.zeros((2,2))  #Consultar, como extendemos la matriz?
        D[0][0] = lam1

        matriz_gen = np.zeros((2,2))
        S = Hv1 * 
    return S, D
        






#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
# D = np.diag([1,0.5,0.25])
# S = np.vstack([
#     np.array([1,-1,1])/np.sqrt(3),
#     np.array([1,1,0])/np.sqrt(2),
#     np.array([1,-1,-2])/np.sqrt(6)
#               ]).T

# A = S@D@S.T
# SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
# assert np.allclose(D,DRH)
# assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# # Pedimos que pase el 95% de los casos
# exitos = 0
# for i in range(100):
#     A = np.random.random((5,5))
#     A = 0.5*(A+A.T)
#     S,D = diagRH(A,tol=1e-15,K=1e5)
#     ARH = S@D@S.T
#     e = normaExacta(ARH-A,p='inf')
#     if e < 1e-5: 
#         exitos += 1
# assert exitos >= 95



