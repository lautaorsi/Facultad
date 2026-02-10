import numpy as np

def calcularAx(matriz,vector):
    res = np.zeros(len(vector))
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            res[i] += (matriz[i][j] * vector[j])
    return res

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

def normaliza(x, p):
    lista_normalizada = []
    for i in range(len(x)):
        ##Calculo la norma de x[i]
        valor_norma = norma(x[i],p)
        ##creo un array del tamaño de x[i]
        array_normalizado = np.zeros((x[i].shape))
        for j in range(len(x[i])): 
            ##normalizo los valores  
            array_normalizado[j] = x[i][j] / valor_norma
        ##agrego el vector normalizado a la lista
        lista_normalizada.append(array_normalizado)     
    return lista_normalizada

def normaMatMC(A, q, p, Np):
    ## Propongo Np vectores de tamaño M  
    xCandidatos = np.random.rand(*(Np,A.shape[1]))
    ##los normalizo
    xCandidatos = normaliza(xCandidatos,p)
    ##calculo ||Ax||p para algun max inicial propuesto
    xMax = xCandidatos[0]
    axMax = norma(calcularAx(A,xMax),q)
    for i in range(len(xCandidatos)):
        ##calculo ||Ax||p para el candidato a max
        aCandidato = norma(calcularAx(A,xCandidatos[i]),q)
        if(axMax <= aCandidato):
            xMax = xCandidatos[i]
            axMax = aCandidato
    return (axMax,xMax)


def normaExacta (A, p=[1,np.inf]) :
    return 0

def condExacto(A,p):    
    return 0

def condMC(A,p):
    return 0


def main():
    assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
    assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
    assert(norma(np.random.rand(10),2)<=np.sqrt(10))
    assert(norma(np.random.rand(10),2)>=0)

    for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
        assert(np.allclose(norma(x,2),1))
    for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
        assert(not np.allclose(norma(x,2),1) )
    for x in normaliza([np.random.rand(k) for k in range(1,11)],np.inf):
        assert( np.allclose(norma(x,np.inf),1) )

    nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
    assert(np.allclose(nMC[0],1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))
    
    nMC = normaMatMC(A=np.eye(2),q=2,p=np.inf,Np=100000)
    assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    
    A = np.array([[1,2],[3,4]])
    nMC = normaMatMC(A=A,q=np.inf,p=np.inf,Np=1000000)
    print(nMC[0])
    assert(np.allclose(nMC[0],normaExacta(A,np.inf),rtol=2e-1)) 
    
    

    return 0


if __name__ == "__main__":
    main()

# 
# Tests L03-Normas
# Tests norma

# 
# Tests normaliza
# Tests normaliza

# 
# Tests normaExacta
# 
# assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
# assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
# assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
# assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
# assert(normaExacta(np.random.random((10,10)),1)<=10)
# assert(normaExacta(np.random.random((4,4)),'inf')<=4)
# 
# Test normaMC
# 
# nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
# assert(np.allclose(nMC[0],1,atol=1e-3))
# assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
# assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))
# 
# nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
# assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
# assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
# 
# A = np.array([[1,2],[3,4]])
# nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
# assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 
# 
# Test condMC
# 
# A = np.array([[1,1],[0,1]])
# A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaMatMC(A,2,2,10000)
# normaA_ = normaMatMC(A_,2,2,10000)
# condA = condMC(A,2,10000)
# assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))
# 
# A = np.array([[3,2],[4,1]])
# A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaMatMC(A,2,2,10000)
# normaA_ = normaMatMC(A_,2,2,10000)
# condA = condMC(A,2,10000)
# assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))
# 
# Test condExacta
# 
# A = np.random.rand(10,10)
# A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaExacta(A,1)
# normaA_ = normaExacta(A_,1)
# condA = condExacta(A,1)
# assert(np.allclose(normaA*normaA_,condA))
# 
# A = np.random.rand(10,10)
# A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaExacta(A,'inf')
# normaA_ = normaExacta(A_,'inf')
# condA = condExacta(A,'inf')
# assert(np.allclose(normaA*normaA_,condA))