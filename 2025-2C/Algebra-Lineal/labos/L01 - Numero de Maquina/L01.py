import matplotlib.pyplot as plt
import math
import numpy as np

l = [-1] * 100

def sucesion3(x):
    if x == 1: return(math.sqrt(2))
    if (l[x-1] != -1): return(l[x-1])

    ans = sucesion3(x-1) * sucesion3(x-1) / math.sqrt(2)
    l[x-1] = ans
    
    return(ans)

def ej3():
    for i in range(1,100):
        l.append(sucesion3(i))


############################################ MODULO ALC ############################################

def matricesIguales(A,B):
    if(A.ndim != B.ndim): return False
    for i in range(A.ndim):
        print(i)
        for j in range(A[i].size - 1) :
            print(j)
            if np.float32(A[i][j]) != np.float32(B[i][j]):
                return False
    return True

def error(x,y):
    return abs(np.float32(y) - x)

def error_relativo(x,y):
    return abs((np.float32(y) - x))  / abs(x)
    

def sonIguales(x,y,atol=1e-08):
    return np.allclose(error(x,y),0,atol=atol)

def main():

    assert(not sonIguales(1,1.1))
    assert(sonIguales(1,1 + np.finfo('float64').eps))
    assert(not sonIguales(1,1 + np.finfo('float32').eps))
    assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
    assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))



    assert(np.allclose(error_relativo(1,1.1),0.1))
    assert(np.allclose(error_relativo(2,1),0.5))
    assert(np.allclose(error_relativo(-1,-1),0))
    assert(np.allclose(error_relativo(1,-1),2))



    assert(matricesIguales(np.diag([1,1]),np.eye(2)))
    assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
    assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))
        
if __name__ == "__main__":
    main()