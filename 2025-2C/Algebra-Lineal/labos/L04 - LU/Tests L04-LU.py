import numpy as np

# Tests L04-LU


def calculaLU(A):
    print(A)
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for i in range(m):
        if np.linalg.det(A) == 0:            ##TODO: CALCULAR DETERMINANTE DE SUS SUBMATRICES PRINCIPALES
            return None, None, None


    L = np.zeros((m,n))
    

    for i in range(m):
        L[i][i] = 1

        
    U = A.copy()

    for i in range(m):
        for j in range(i+1,m):
            if U[j][i] != 0: 
                cant_op += 1 + (2* (m - (i+1)))
                if U[i][i] == 0:
                    return None, None, None
                Mi = (U[j][i] / U[i][i])
                U[j] = U[j] -  ( Mi * U[i])
                L[j][i] = Mi
                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    print(L)
    print(U)
    print(cant_op)
    return L, U, cant_op

def res_tri(L,b,inferior=True):
    print("asd")
    filasL = L.shape[0]
    res = np.zeros(filasL)
    if inferior:
        for i in range(filasL):
            res[i] = b[i]
            for j in range(i):
                res[i] -= L[j] * res[j] 
    else:

        for i in range(filasL, 0):
            
            res[i] = b[i]
            for j in range(i):
                res[i] -= L[j] * res[j]     



def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0] # filas
    n=A.shape[1] # columns
    Ac = A.copy()

    if m!=n:
        print('Matriz no cuadrada')
        return

    for k in range(0, n):
      if Ac[k][k] != 0:
        for i in range(k+1, n):
          gama = Ac[i][k] / Ac[k][k]
          cant_op += 1
          for j in range(k, m):
            Ac[i][j] = Ac[i][j] - (gama * Ac[k][j])
            cant_op += 2
          Ac[i][k] = gama
      else:
        raise Exception('No se puede dividir entre cero')

    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre
    ## la matriz Ac
    # print(Ac)
    L = np.tril(Ac, -1) + np.eye(n)
    U = np.triu(Ac)

    return L, U, cant_op

def multiplicacion_matricial(A, B):
    A = np.array(A, dtype=float)  # asegurar numpy array
    B = np.array(B, dtype=float)

    # chequeo dimensiones
    if A.shape[1] != B.shape[0]:
        return None

    n, m = A.shape       # A es n×m
    m2, p = B.shape      # B es m×p

    # Creo la matriz resultado
    C = np.zeros((n, p))

    # Triple bucle: filas de A, columnas de B, suma sobre productos
    for i in range(n):         # filas de A
        for j in range(p):     # columnas de B
            s = 0.0
            for k in range(m): # columnas de A = filas de B
                s += A[i, k] * B[k, j]
            C[i, j] = s

    return C


def calcula_LU(A):
  '''Calcula la factorizacion LU de la matriz A y retorna las matrices L y U, junto con el numero de operaciones realizadas. En caso de que la amtriz no pueda factorizarse retorna None '''

  cant_op = 0
  m=A.shape[0]
  n=A.shape[1]
  Ac = A.copy()

  if m!=n:
      print('Matriz no cuadrada')
      return

  for k in range(0, n):
    if Ac[k][k] != 0:
      for i in range(k+1, n):
        gama = Ac[i][k] / Ac[k][k]
        cant_op += 1
        for j in range(k, m):
          Ac[i][j] = Ac[i][j] - (gama * Ac[k][j])
          cant_op += 2
        Ac[i][k] = gama
    else:
      raise Exception('No se puede dividir entre cero')

  L = np.tril(Ac, -1) + np.eye(n)
  U = np.triu(Ac)

  return L, U, cant_op

def res_tri(L, b, inferior=True):
  ''' Resuelve el sistema Lx = b, donde L es triangular. Si inferior es True, L es triangular inferior, si es False, L es triangular superior. '''

  n = L.shape[0]
  x = np.zeros(n) # Creo el vector solucion

  if inferior:
    for i in range(n):
      suma = np.dot(L[i, :i], x[:i]) # fila i desde la columna 0 hasta la columna i-1
      x[i] = (b[i] - suma) / L[i, i] # en suma tenemos ya los valores de los x anteriores calculados con mis nuevas variables en esta fila
  else: # tengo que volver sobre esto para poder explicarlo con crayones
    for i in range(n-1, -1, -1): # en este caso es para atras el recorrido
      suma = np.dot(L[i, i+1:], x[i+1:]) # fila i, desde la columna i+1 hasra el final
      x[i] = (b[i] - suma) / L[i, i]

  return x

def inversa(A):
  ''' Calcla la inversa de A empleando la factorizacion LU y las funciones que resuelven sistemas triangulares.'''

  # Cuando invertimos una matriz usando la identidad de un lado y triangulando, lo que estamos resolviendo es n sistemas lineales
  # del tipo A*x_j = e_j donde e_j es la columna j de la identidad
  # y cada solucion x_j sera una columna de A^-1. Como A la puedo escribir como LU entonces nos quedaria LU*x_j=e_j

  n = A.shape[0]
  L, U, _ = calcula_LU(A)
  A_inv = np.zeros((n,n), dtype=float)

  # recorremos cada columna de la identidad
  for col_id_j in range(n):
      e_j = np.eye(n)[:, col_id_j]              # columna j de la identidad
      y = res_tri(L, e_j, inferior=True)        # Ly = e_j
      x = res_tri(U, y, inferior=False)         # Ux = y
      A_inv[:, col_id_j] = x                    # guardo la columna en A_inv

  return A_inv

def calculaLDV(A):
  ''' Calcula la factorizacion LDV de la matriz A, de forma tal que A=LDV con L triangular inferior, D diaogonal y V triangular superior. Retorna las matrices L, D y V. Si no puede factorizarse retorna None'''

  L, U, _ = calcula_LU(A)
  Vt, D, _ = calcula_LU((U.T))
  V = (Vt).T
  return L, D, V


def main():
    # Test LDV:

    L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
    D0 = np.diag([1,2,3])
    V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
    A =  L0 @ D0  @ V0
    L,D,V = calculaLDV(A)
    print(L,D,V)
    assert(np.allclose(L,L0))
    assert(np.allclose(D,D0))
    assert(np.allclose(V,V0))

    L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
    D0 = np.diag([3,2,1])
    V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
    A =  L0 @ D0  @ V0
    L,D,V = calculaLDV(A)
    assert(np.allclose(L,L0,1e-3))
    assert(np.allclose(D,D0,1e-3))
    assert(np.allclose(V,V0,1e-3))


if __name__ == "__main__":
    main()
    
    

# # Test inversa

# ntest = 10
# iter = 0
# while iter < ntest:
#     A = np.random.random((4,4))
#     A_ = inversa(A)
#     if not A_ is None:
#         assert(np.allclose(np.linalg.inv(A),A_))
#         iter += 1

# # Matriz singular devería devolver None
# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# assert(inversa(A) is None)





# # Tests SDP

# L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
# D0 = np.diag([1,1,1])
# A = L0 @ D0 @ L0.T
# assert(esSDP(A))

# D0 = np.diag([1,-1,1])
# A = L0 @ D0 @ L0.T
# assert(not esSDP(A))

# D0 = np.diag([1,1,1e-16])
# A = L0 @ D0 @ L0.T
# assert(not esSDP(A))

# L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
# D0 = np.diag([1,1,1])
# V0 = np.array([[1,0,0],[1,1,0],[1,1+1e-10,1]]).T
# A = L0 @ D0 @ V0
# assert(not esSDP(A))