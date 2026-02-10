import numpy as np
# ------------- Aux --------------
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

def es_simetrica(A, umbral=1e-10):
    return np.all(np.abs(A - A.T) < umbral)

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

def norma_uno(A):
  acum = 0
  for c in range(A.shape[1]):
    sum_abs = 0
    for r in range(A.shape[0]):
      sum_abs += abs(A[r][c])
    acum = max(acum, sum_abs)
  return acum

def norma_inf(A):
  acum = 0
  for r in range(A.shape[0]):
    sum_abs = 0
    for c in range(A.shape[1]):
      sum_abs += abs(A[r][c])
    acum = max(acum, sum_abs)
  return acum

def norma_inf_vec(v):
  acum = 0
  for i in range(len(v)):
    acum = max(acum, abs(v[i]))
  return acum

def norma_dos(vec):
    return (sum(i**2 for i in vec ))**0.5

def matriz_identidad(m):
   I = []
   for i in range(m):
      fila = []
      for j in range(m):
         if i == j:
            fila.append(1)
         else:
            fila.append(0)
      I.append(fila)
   return I

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def transponer(M):
    """
    Devuelve la matriz transpuesta de M como un np.ndarray.
    Acepta M como lista de listas o como np.ndarray.
    """
    M = np.array(M, dtype=float)   # asegurar array
    filas, columnas = M.shape
    T = np.zeros((columnas, filas), dtype=float)

    for i in range(filas):
        for j in range(columnas):
            T[j, i] = M[i, j]

    return T

def dot(vec1, vec2):
    v1 = np.array(vec1, dtype=float).reshape(-1)
    v2 = np.array(vec2, dtype=float).reshape(-1)

    if v1.shape != v2.shape:
        print("dot: deben ser del mismo tamaño")
        return None

    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def vector_householder(x, tol=1e-12):
    # x es un vector (1D numpy array)
    normx = norma_dos(x)
    if normx <= tol:
        return None  # no se puede construir reflector útil

    # Elegimos el signo para evitar cancelación numérica
    alpha = sign(x[0]) * normx

    # Vector v = x + alpha e1
    v = x.copy()
    v[0] = v[0] + alpha

    # Si v es casi cero, no sirve
    normv = norma_dos(v)
    if normv <= tol:
        return None

    return v  # no normalizamos aquí

# +++++++++ funciones de Ivan ++++++++++
def traspuesta(A):
    A = np.array(A)
    n, m = A.shape
    B = np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i, j]
    return B

# ---------- Funciones pedidas -----------

# -- Labo 0 --
# TODO 

# -- Labo 1 --
def error(x,y):
   ''' Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64 '''

def error_relativo(x,y):
   ''' Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float 64'''

def matrices_iguales(A,B):
   ''' Devuelve True si ambas matrices son iguales y False en otro caso, considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores'''
   if A.shape != B.shape: #si no tienen misma dimension no son iguales
      return False
   
   n,m = A.shape # ok tienen misma dimension, ahora vemos valor por valor si son iguales
  
   for i in range(n):
      for j in range(m):
         if A[i,j] != B[i,j]:
            return False
   return True

# -- Labo 2 --
def rota(theta):
    """Recibe un angulo theta y retorna una matriz de 2x2 que rota un vector dado en un angulo theta"""
    R = np.array([
       [np.cos(theta), -np.sin(theta)],
       [np.sin(theta), np.cos(theta)]
    ])
    return R

def escala(s):
    """Recibe una tira de numeros s y retorna una matriz cuadrada de n x n, donde n es el tamano de s.
    La matriz escala la componente i de un vector de Rn en un factor s[i]"""
    n = len(s)
    S = np.zeros((n,n))
    for i in range(n):
       S[i,i] = s[i]
    return S

def rota_y_escala(theta, s):
    """Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s"""
    # TODO consultar esta bien aca retornar None? 
    if len(s) != 2:
       return None
    R = rota(theta)
    S = escala(s)
    return multiplicacion_matricial(S, R)

def afin(theta, s, b):
    """Recibe un angulo theta , una tira de numeros s (en R2) , y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b"""
    if len(s) != 2 or len(b) != 2:
       return None 
    R_S = rota_y_escala(theta, s)
    bx, by = b[0], b[1]
    A = np.array([
       [R_S[0,0], R_S[0,1], bx],
       [R_S[1,0], R_S[1,1], by],
       [0.0, 0.0, 1.0]
    ], dtype=float)
    return A 
    
def trans_afin(v, theta, s, b):
    """Recibe un vector v (en R2) , un angulo theta ,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a v"""
    # la funcin 'afin' nos retorna una matriz de 3x3 -> hay que modificar v que es de R2
    v_aux = [v[0], v[1], 1.0] # le enchufo un 1 al final
    A = afin(theta, s, b)
    if A is None:
       return None
    w_aux = multiplicacion_matricial(A, v_aux)
    w = [w_aux[0], w_aux[1]] 
    return w

# -- Labo 3 -- 

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    Ejemplos de p:
        p = 1  → norma 1
        p = 2  → norma Euclidea
        p = 'inf' → norma infinito
    """
    if p == 1:
       return sum(abs(i) for i in x) 
    if p == 2:
       return norma_dos(x)
    if p == 'inf':
       return norma_inf_vec(x)
    return None

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacíos, y un escalar p.
    Devuelve una lista donde cada elemento corresponde a normalizar
    los elementos de X con la norma p.
    """
    result = []
    for vec in X:
       vec_norm = norma(vec, p)
       if vec_norm == 0:
          result.append(vec.copy())
       else:
          vec_normalizado = [elem / vec_norm for elem in vec]
          result.append(vec_normalizado)
    return result

def normaMatMC(A, q, p, Np):
    """
    Aproxima la norma inducida ||A||_{q,p} por Monte Carlo.
    
    Parámetros:
        A  : matriz (np.ndarray de m×n)
        q  : norma para ||Ax||
        p  : norma para ||x||
        Np : número de muestras
        
    Retorna:
        norma_aprox : aproximación del máximo
        x_max       : vector donde se alcanza el máximo (normalizado en norma p)
    """
    # Asegurar array float
    A = np.array(A, dtype=float)
    m, n = A.shape

    best_val = -1.0
    best_x = None

    for _ in range(Np):
        # Generar vector aleatorio en [-1,1]^n
        x = np.random.uniform(-1, 1, size=n)

        # Normalizar en norma p
        nx = norma(x, p)
        if nx == 0:
            continue
        x = x / nx

        # Calcular Ax
        y = multiplicacion_matricial(A, x)

        # Calcular norma q
        val = norma(y, q)

        # ¿Mejora?
        if val > best_val:
            best_val = val
            best_x = x.copy()

    # Si nunca encontramos un vector válido
    if best_x is None:
        return 0.0, np.zeros(n)

    return best_val, best_x

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    
    Si p incluye más normas, extender la funcionalidad.
    """
    result = []
    if 1 in p:
       result.append(norma_uno(A))
    if 'inf' in p :
       result.append(norma_inf(A))
    # if len(p) > 2:
       # TODO hacemos otras normas, tipo p?
    return result    

def condMC(A, p):
    #TODO si no es inversible, que retorno?
    """
    Devuelve el número de condición de A usando la norma inducida p,
    aproximando la norma mediante el método Monte Carlo.
    """
    normaA, _ = normaMatMC(A, p, p, 10)
    normaInv, _ = normaMatMC(inversa(A), p, p, 10)
    return normaA * normaInv

def condExacto(A, p):
    """
    Devuelve el número de condición de A a partir de la fórmula
    cond(A) = ||A||_p * ||A^{-1}||_p
    usando la norma p.
    """
    normA = normaExacta(A, [p])
    normInv = normaExacta(inversa(A), [p])
    return normA[0] * normInv[0]

# -- Labo 4 -- 

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

def calcula_LDV(A):
  ''' Calcula la factorizacion LDV de la matriz A, de forma tal que A=LDV con L triangular inferior, D diaogonal y V triangular superior. Retorna las matrices L, D y V. Si no puede factorizarse retorna None'''

  L, U, _ = elim_gaussiana(A)
  D = np.diag(np.diag(U)) # el primer np.diag toma la diagonal de la matriz U que es t.s en forma de vector, luego si le hacemos np.diag(vector) tenemos una matriz diagonal con todos ceros y ese vector en la diagonal
  V = multiplicacion_matricial(inversa(D), U)
  return L, D, V

def esSDP(A, atol=1e-8):
  ''' Chequea si la matriz A es simetrica definida positiva (SDP) usando la factorizacion LDV. Retorna True si es SDP, False en caso contrario.'''

  if not es_simetrica(A, atol):
    return False
  L, D, V = calcula_LDV(A)
  print("L", L)
  print("D", D)
  return np.all(np.linalg.eigvals(D) >= 0)

# -- Labo 5 -- 

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    ''' A una matriz de nxn
        tol la tolerancia con la que se filtran elementos nulos en R 
        retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
        retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
        Si la matriz A no es de nxn, debe retornal None'''
    if A.shape[0] != A.shape[1]:
        print("La matriz A debe ser nxn para usar QR_con_GS.")
        return None
    # nos aseguramos de trabajar en float para evitar truncamientos enteros
    A = A.astype(float, copy=False)
    n = A.shape[0]
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    n_ops = 0
    for j in range(n):
        # Tomamos la primer columna de A
        q = A[:, j]
        for k in range(j): # el primero no se ejecuta, entonces k < j -> computamos la proyeccion sobre q_k (las ya ortogonales de Q)
            # Restamos las proyecciones sobre q_k anteriores
            R[k,j] = dot(Q[:, k], q) # coeficientes de proyección
            n_ops += (2*n) - 1 
            q = q - R[k,j] * Q[:, k]    # quitar componente en direccion q_k
            # Al actualizar el residual inmediatamente reducimos la acumulacion del error numerico, mucho mas estable que el CGS (el clasico)
            n_ops += 2*n
        # Normalizamos
        # guardo la norma en la matriz triangular superior
        R[j,j] = norma_dos(q)
        n_ops += n + 1
        if R[j,j] <= tol:
           # columna dependientem definimos
           Q[:, j] = 0
           R[j,j] = 0.0
        else:
           Q[:, j] = q / R[j,j]
        n_ops += 2*n
    if retorna_nops:
        return Q, R, n_ops
    else:
        return Q, R

def QR_con_GS_rect(A, tol=1e-12):
    A = np.array(A, dtype=float)
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = norma_dos(v)
        if R[j, j] < tol:
            return None

        Q[:, j] = v / R[j, j]

    return Q, R

def QR_con_HH(A, tol=1e-12):
    A = A.astype(float, copy=True)   # copiamos para no modificar la original
    m, n = A.shape

    # if m < n:
    #     print("A debe tener m >= n")
    #     return None

    R = A.copy()
    Q = np.eye(m)

    # Para cada columna k
    for k in range(n):
        # Tomamos el subvector x desde la fila k hasta el final
        x = R[k:, k].copy()

        # Construimos el vector de Householder v
        v = vector_householder(x, tol)
        if v is None:
            continue  # no hay nada que reflejar

        # Calculamos v^T v una sola vez
        vTv = 0.0
        for val in v:
            vTv += val * val

        # --- Aplicamos reflejo a R ---
        # Solo afecta a las filas k: y columnas k:
        for j in range(k, n):
            # s = v^T * R[k:, j]
            s = 0.0
            for i in range(len(v)):
                s += v[i] * R[k+i, j]

            # R[k:, j] = R[k:, j] - 2 * v * s / (v^T v)
            coef = 2.0 * s / vTv
            for i in range(len(v)):
                R[k+i, j] = R[k+i, j] - coef * v[i]

        # --- Aplicamos reflejo a Q ---
        # Afecta todas las columnas de Q, pero solo filas k:
        for j in range(m):
            s = 0.0
            for i in range(len(v)):
                s += v[i] * Q[k+i, j]
            coef = 2.0 * s / vTv
            for i in range(len(v)):
                Q[k+i, j] = Q[k+i, j] - coef * v[i]

    # Limpiar pequeños errores numéricos debajo de diagonal
    for i in range(1, m):
        # solo iterar hasta el número de columnas (n), porque R es m x n (m >= n)
        for j in range(min(i, n)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0

    return Q.T, R    # Devolvemos Q ortogonal (transpuesto porque fuimos aplicando desde la izquierda)

def calculaQR(A, metodo='RH', tol=1e-12, retorna_nops=False):
    """
    A: matriz de nxn
    tol: tolerancia para filtrar elementos pequeños
    metodo: 'RH' para Householder, 'GS' para Gram-Schmidt
    retorna_nops (solo afecta GS): si True, devuelve también el número de operaciones

    Retorna: (Q, R) o (Q, R, n_ops)
    Si el método no es válido → None
    """
    
    # Validar método
    if metodo not in ['RH', 'GS']:
        print("Método inválido. Use 'RH' o 'GS'.")
        return None
    
    # Si el método es Gram-Schmidt
    if metodo == 'GS':
        # Pasamos retorna_nops solo si el usuario lo pide
        return QR_con_GS(A, tol=tol, retorna_nops=retorna_nops)
    
    # Si el método es Householder, no usa n_ops
    if metodo == 'RH':
        # Householder no retorna n_ops, aunque se lo pidan
        return QR_con_HH(A, tol=tol)

# -- Labo 6 -- 

def metpot2k(A, tol=1e-15, K=1000):
    """
    Método de la potencia con 2 vectores (versión mejorada).
    
    Parámetros:
        A   : matriz de n x n
        tol : tolerancia en la diferencia entre dos estimaciones sucesivas del autovector
        K   : número máximo de iteraciones a ejecutar
    
    Retorna:
        v : autovector aproximado (normalizado)
        λ : autovalor dominante aproximado
        k : número de iteraciones realizadas
    """
    # vector inicial no nulo
    n = A.shape[0]
    x = np.random.uniform(-1, 1, size=n)
    it = 0 
    while it < K:
       x_old = x.copy()
       w = multiplicacion_matricial(A, x.reshape(n,1))
       w = w.reshape(n)
       if norma(w,2) != 0:
          x = w / norma(w, 2)
       if norma( x - x_old, 2) < tol:
          break
       
       it += 1
    # λ≈xkT​Axk​
    Ax = multiplicacion_matricial(A, x.reshape(n,1))
    lamb = multiplicacion_matricial(x.reshape(1,n), Ax.reshape(n,1))
    return x, lamb[0][0], it

def diagRH(A, tol=1e-15, K=1000):
    """
    Diagonalización usando reflexiones de Householder.
    """
    # defensa: trabajar sobre copia float y forzar simetría numérica
    A = np.array(A, dtype=float, copy=True)
    # forzar simetría para evitar falsos negativos por errores de punto flotante
    A = (A + A.T) / 2.0

    n = A.shape[0]
    if A.shape[1] != n:
        return None
    # usar np.allclose con una tolerancia razonable para chequear simetría
    if not np.allclose(A, A.T, atol=max(tol, 1e-12)):
        return None

    if n == 1:
       S = np.array([[1.0]])
       D = np.array([[A[0,0]]], dtype=float)
       return S, D

    v1, lamb1, _ = metpot2k(A, tol, K)
    v = vector_householder(v1, tol)
    if v is None:
       H = np.eye(n, dtype = float)
    else:
       v = v.reshape(n,1)
       I = np.eye(n)
       vvT = multiplicacion_matricial(v, v.T)
       norma_v = norma(v.reshape(n), 2)
       if norma_v == 0:
          H = np.eye(n, dtype = float)
       else:
          beta = 2.0 / (norma_v**2)
          Hv = vvT.copy()
          for i in range(n):
            for j in range(n):
                Hv[i, j] = beta * Hv[i, j]
          H = I - Hv

    HA = multiplicacion_matricial(H, A)
    B = multiplicacion_matricial(HA, transponer(H))
    lamb1 = B[0,0]
    sub_A = B[1:n, 1:n].copy()

    res = diagRH(sub_A, tol, K)
    if res is None:
        # mejor devolver información de diagnóstico en lugar de None silencioso
        print("diagRH: fallo recursivo en sub_A.shape =", sub_A.shape)
        return None
    S_tilde, D_tilde = res

    D_final = np.zeros((n,n), dtype=float)
    D_final[0,0] = lamb1
    D_final[1:, 1:] = D_tilde
    S_block = np.eye(n, dtype=float)
    S_block[1:, 1:] = S_tilde
    S_final = multiplicacion_matricial(H, S_block)
    return S_final, D_final

# -- Labo 7 -- 

def transiciones_azar_continuas(n): # todas las entradas > 0 -> densas
    """
    n: cantidad de filas (y columnas) de la matriz de transición.
    
    Retorna una matriz T de n x n normalizada por columnas,
    con entradas aleatorias en el intervalo [0, 1].
    """
    matriz_aleatoria = np.random.uniform(0.0, 1.0, (n,n))
    for j in range(n):
       suma_col = np.sum(matriz_aleatoria[:, j])
       if suma_col != 0:
          matriz_aleatoria[:, j] = matriz_aleatoria[:, j] / suma_col
       else:
          matriz_aleatoria[:, j] = np.ones(n) / n
    return matriz_aleatoria          
    
def transiciones_azar_uniforme(n, thres): # refleja sistemas dispersos -> pocas conexiones
    """
    n     : cantidad de filas (y columnas) de la matriz de transición.
    thres : probabilidad de que una entrada sea distinta de cero.
    
    Retorna una matriz T de n x n normalizada por columnas.
    
    El elemento (i, j) es distinto de cero si el número aleatorio generado
    para (i, j) es <= thres. Todos los elementos no nulos de una columna
    j serán iguales entre sí, y valdrán 1 dividido por la cantidad de elementos
    distintos de cero en dicha columna.
    """
    T = np.zeros((n,n))
    M = np.random.uniform(0, 1, (n, n))
    mask = np.where(M <= thres, 1, 0)
    for j in range(n):
       k = np.sum(mask[:, j])
       if k > 0:
          T[:, j] = mask[:, j] / k
       else:
          T[:, j] = np.ones(n) / n
    return T

def nucleo(A, tol=1e-15):
    """
    A   : matriz de m x n
    tol : tolerancia para considerar que un autovalor es cero.
    
    Calcula el núcleo de la matriz A diagonalizando la matriz (A^T) * A
    usando el método diagRH. 
    
    El núcleo corresponde a los autovectores asociados a autovalores
    cuyo módulo sea <= tol.
    
    Retorna una matriz de n x k cuyos k vectores columna son los autovectores
    que forman una base del núcleo.
    """
    m, n = A.shape
    A = np.array(A, dtype=float)
    AT = transponer(A)
    ATA = multiplicacion_matricial(AT, A) 
    diag_result = diagRH(ATA, tol)
    if diag_result is None:
        return None
    S, D = diag_result
    autovalores = np.diag(D)
    indices = np.where(np.abs(autovalores) <= tol)[0]
    if indices.size == 0:
        return np.zeros((n,0), dtype= float)
    else:
       S = np.array(S, dtype = float)
       N = S[:, indices].copy()
       return N
       
def crear_ala(listado, m_filas, n_columnas, tol=1e-15):
    """
    listado : lista con tres elementos: 
                - lista de índices i
                - lista de índices j
                - lista de valores A_ij
    m_filas    : cantidad de filas de la matriz
    n_columnas : cantidad de columnas de la matriz
    tol        : valores con módulo menor a tol deben descartarse
    
    Idealmente, 'listado' incluye solo posiciones de valores no nulos.
    
    Retorna:
        - Un diccionario {(i, j): A_ij} con los elementos no nulos de la matriz A
        - Una tupla (m_filas, n_columnas) con las dimensiones de A
    """
    if not (len(listado[0]) == len(listado[1]) == len(listado[2])):
       return None
    diccionario = dict()
    for i in range(len(listado[0])):
       if abs(listado[2][i]) > tol:
          diccionario[(listado[0][i], listado[1][i])] = listado[2][i] 
    return diccionario, (m_filas, n_columnas)
    
def multiplicar_ala_vector(A, v):
    """
    A : matriz rala (representada con el diccionario creado por crear_ala)
    v : vector
    
    Retorna el vector w resultado de multiplicar A por v.
    """
    diccionario, tupla = A
    w = np.zeros(tupla[0])
    for (i,j), aij in diccionario.items():
       w[i] += aij * v[j] 
    return w 

# -- Labo 8 -- 

def svd_reducida(A, k="max", tol=1e-15):
    A = np.array(A, dtype=float)
    m, n = A.shape

    if m >= n:
       AT = (A).T
       ATA = (AT@ A)
       diag_result = diagRH(ATA, tol)
       linalg = np.linalg.eig(ATA)
       if diag_result is None:
           return np.zeros((m,0)), np.array([]), np.zeros((n,0))
       V_all, D = diag_result
       autovalores = np.array(np.diag(D), dtype=float)

       # proteger contra pequeñas negatividades numéricas
       autovalores = np.clip(autovalores, 0.0, None)

       # ordenar de mayor a menor
       order = np.argsort(autovalores)[::-1]
       autovalores = autovalores[order]
       V_all = np.array(V_all, dtype=float)[:, order]

       # calcular todas las sigmas y seleccionar por sigma > tol
       sigmas_all = np.sqrt(autovalores)

        # limpieza adicional por ruido numérico
        # si autovalor < tol → sigma a cero
       sigmas_all = np.where(autovalores > tol, sigmas_all, 0.0)

       if k == "max":
           keep_idx = np.where(sigmas_all > tol)[0]
       else:
           k = int(k)
           keep_idx = np.arange(min(k, len(sigmas_all)))
           
       if keep_idx.size == 0:
          return np.zeros((m,0)), np.array([]), np.zeros((n,0))

       V_reducida = V_all[:, keep_idx].copy()

       # asegurar normalizacion de columnas
       for i in range(V_reducida.shape[1]):
           nv = norma_dos(V_reducida[:, i])
           if nv > 0:
               V_reducida[:, i] = V_reducida[:, i] / nv

       sigmas = sigmas_all[keep_idx]

       # proteger contra NaN/inf
       if not np.all(np.isfinite(sigmas)):
           sigmas = sigmas[np.isfinite(sigmas)]
           if sigmas.size == 0:
               return np.zeros((m,0)), np.array([]), np.zeros((n,0))
           V_reducida = V_reducida[:, :len(sigmas)]

       AV = np.array((A@V_reducida))
       U = np.zeros((m, len(sigmas)))
       for i in range(len(sigmas)):
           if sigmas[i] > tol:
               U[:, i] = AV[:, i] / sigmas[i]
           else:
               U[:, i] = 0.0

       # normalizar U por si acaso
       for i in range(U.shape[1]):
           nu = norma_dos(U[:, i])
           if nu > 0:
               U[:, i] = U[:, i] / nu
  
       return U, sigmas, V_reducida

    else:
        AAT = A@(A.T)
        diag_result = diagRH(AAT, tol)
        if diag_result is None:
            return np.zeros((m,0)), np.array([]), np.zeros((n,0))
        U_all, D = diag_result
        autovalores = np.array(np.diag(D), dtype=float)

        autovalores = np.clip(autovalores, 0.0, None)

        order = np.argsort(autovalores)[::-1]
        autovalores = autovalores[order]
        U_all = np.array(U_all, dtype=float)[:, order]

        sigmas_all = np.sqrt(autovalores)

        # limpieza adicional por ruido numérico
        # si autovalor < tol → sigma a cero
        sigmas_all = np.where(autovalores > tol, sigmas_all, 0.0)

        if k == "max":
            keep_idx = np.where(sigmas_all > tol)[0]
        else:
            k = int(k)
            keep_idx = np.arange(min(k, len(sigmas_all)))

        if keep_idx.size == 0:
            return np.zeros((m,0)), np.array([]), np.zeros((n,0))

        U_reducida = U_all[:, keep_idx].copy()

        for i in range(U_reducida.shape[1]):
            nu = norma_dos(U_reducida[:, i])
            if nu > 0:
                U_reducida[:, i] = U_reducida[:, i] / nu

        sigmas = sigmas_all[keep_idx]
        if not np.all(np.isfinite(sigmas)):
            sigmas = sigmas[np.isfinite(sigmas)]
            if sigmas.size == 0:
                return np.zeros((m,0)), np.array([]), np.zeros((n,0))
            U_reducida = U_reducida[:, :len(sigmas)]

        AT = (A).T
        ATU = np.array(AT@U_reducida)
        V = np.zeros((n, len(sigmas)))
        for i in range(len(sigmas)):
            if sigmas[i] > tol:
                V[:, i] = ATU[:, i] / sigmas[i]
            else:
                V[:, i] = 0.0

        # normalizar V
        for i in range(V.shape[1]):
            nv = norma_dos(V[:, i])
            if nv > 0:
                V[:, i] = V[:, i] / nv

        return U_reducida, sigmas, V