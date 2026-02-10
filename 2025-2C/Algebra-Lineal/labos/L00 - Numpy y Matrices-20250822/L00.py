import numpy as np
import copy

def esCuadrada(matriz):
    for fila in matriz:
        if fila.size != len(matriz):    ## Si la cantidad de elementos en la fila es distinta a la cantidad de filas 
            return False                ## entonces no es cuadrada
    return True



def triangSup(matriz):
    if  not(esCuadrada(matriz)):        ## Para calcular la matriz triangular necesito que sea cuadrada
        return "La matriz no es cuadrada"
    
    for i in range(len(matriz)):
        for j in range(i + 1):
            matriz[i][j] = 0
    return matriz

def triangInf(matriz):
    if not(esCuadrada(matriz)):         ## Para calcular la matriz triangular necesito que sea cuadrada
        return "La matriz no es cuadrada"
    
    for i in range(len(matriz)):
        for j in range(i + 1):
            matriz[j][i] = 0
    return matriz

def diagonal(matriz):
    if not(esCuadrada(matriz)):         ## Para la diagonal necesito que sea cuadrada
        return "La matriz no es cuadrada"
    for i in range(len(matriz)):
        for j in range(matriz[i].size):
            if i != j:
                matriz[i][j] = 0

def traza(matriz):
    if not(esCuadrada(matriz)):         ## Para calcular la traza necesito que sea cuadrada
        return "La matriz no es cuadrada"

    res = 0
    for i in range(len(matriz)):
        for j in range(matriz[i].size):
            if i == j:
                res += matriz[i][j]
    return res

def traspuesta(matriz):
    nueva_matriz = copy.deepcopy(matriz)
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            print(f"{i} {j} {matriz[i][j]}")
            nueva_matriz[j][i] = matriz[i][j]
    return nueva_matriz

def esSimetrica(matriz):
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if matriz[i][j] != matriz[j][i]:
                return False
    return True

def calcularAx(matriz,vector):
    res = np.zeros(len(vector))
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            res[i] += (matriz[i][j] * vector[j])
    return res

def intercambiarFilas(matriz, indiceF1, indiceF2):
    matriz[indiceF1] += matriz[indiceF2] 
    matriz[indiceF2] = matriz[indiceF1] - matriz[indiceF2] 
    matriz[indiceF1] -= matriz[indiceF2]
    return matriz

def sumar_fila_multiplo(matriz,indiceF1,indiceF2,escalar):
    for k in range(len(matriz[indiceF1])):
        matriz[indiceF1][k] += matriz[indiceF2][k] * escalar
    return matriz

def esDiagonalmenteDominante(matriz):
    for i in range(len(matriz)):
        sumatoria_abs = 0
        
        for j in range(len(matriz[i])):
            if i != j:
                sumatoria_abs += abs(matriz[i][j]) 

        if abs(matriz[i][i] <= sumatoria_abs):
            return False

    return True    

def matrizCirculante(vector):
    matriz = np.zeros((len(vector),len(vector)))
    for i in range(len(vector)):
        matriz[i] = np.roll(vector, shift=i)
    return matriz

def matrizVandermonde(vector):
    matriz = np.zeros((len(vector),len(vector)))
    for i in range(len(vector)):
        for j in range(len(vector)):
            matriz[i][j] = pow(vector[j], (i - 1))
    return matriz 

def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    return (fibonacci(n - 1) + fibonacci(n - 2)) 

def matrizFibonacci(n):
    matriz = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matriz[i][j] = fibonacci(i+j)
    return matriz

def matrizHilbert(n):
    matriz = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matriz[i][j] = 1/(i+j+1)


def main():
    a_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
    a_vector = np.array([1.,2.,3.])
    print(matrizFibonacci(5))
    return 0



if __name__ == '__main__':
    main()