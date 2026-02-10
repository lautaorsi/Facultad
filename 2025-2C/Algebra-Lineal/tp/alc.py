import numpy as np
import os

# Grupo HouseOfVectors TN

# Primero estan los 5 ejercicios del TP
# Despues todos los modulos
# Por último las auxiliares que necesitamos para los ejercicios

# Ejercicio 1:

def cargarDataset(carpeta):
    xt , yt = extraer_dato_de(carpeta,"train")
    xv , yv = extraer_dato_de(carpeta,"val")
    return xt , yt , xv , yv

def extraer_dato_de(carpeta,tipo):
    #Carga los embeddings de cats y dogs para train y val
    path = os.path.join(carpeta, tipo)

    rutaCats = os.path.join(path, "cats", "efficientnet_b3_embeddings.npy")
    embeddingsCats = np.load(rutaCats)
    xt = traspuesta(embeddingsCats)
    largo_embedding_cats = embeddingsCats.shape[1]
    yt = np.array([np.ones(largo_embedding_cats), np.zeros(largo_embedding_cats)]) # [1, 0]t

    rutaDogs = os.path.join(path, "dogs", "efficientnet_b3_embeddings.npy")
    embeddingsDogs = np.load(rutaDogs)
    xt2 = traspuesta(embeddingsDogs)
    xt = concatenar_filas(xt,xt2) 
    largo_embedding_dogs = embeddingsDogs.shape[1]
    yt2 = np.array([np.zeros(largo_embedding_dogs), np.ones(largo_embedding_dogs)]) #[0, 1]t
    
    yt = concatenar_columnas(yt,yt2) 
    
    return xt,yt

# Ejercicio 2:

def pinvEcuacionesNormales(X, L, Y):
    # X (nxp) matriz de embeddings
    # L matriz de Cholesky 
    # Y (mxp) matriz de targets de entrenamiento

    n, p = X.shape
    
    # Calculamos W = Y * X⁺ según caso
    if (rango(X) == p and n > p):

        # Caso a. Resuelvo el sistema (X^t * X) U = X^t
        # Tengo L tal que L * L^t = X^t * X
        # Reemplazando eso tengo L L^t U = X^t.

        # Llamo Z a L^t U. Entonces me queda L Z = X^t.
        # Como L es triangular inferior lo resuelvo con forward_substitution
        Z = res_tri_matricial(L, traspuesta(X), inferior=True)

        # Ese Z era L^t U. Y como L^t es triangular superior lo resuelvo con backward_substitution
        U = res_tri_matricial(traspuesta(L), Z, inferior=False)

        #Finalmente tengo Y y U para multiplicarlas y obtener W
        W = multiplicar_matrices(Y, U)
        
    elif (rango(X) == n and n < p):

        # Caso b. Resuelvo el sistema V * (X * X^t) = X^t
        # Tengo L tal que L * L^t = X * X^t
        # Reemplazando eso tengo V (L L^t) = X^t.

        #Transpongo en ambos lados (V (L L^t))^t = (X^t)^t
        # Queda (L L^t)^t V^t = X
        # L L^t V^t = X

        # Ahora hago lo mismo que en el anterior caso
        # Llamo Z a L^t V^t. Entonces me queda L Z = X
        # Como L es triangular inferior lo resuelvo con forward_substitution
        Z = res_tri_matricial(L, X, inferior=True)
        
        # Ese Z era L^t V^t. Y como L^t es triangular superior lo resuelvo con backward_substitution
        Vt = res_tri_matricial(traspuesta(L), Z, inferior=False)

        V = traspuesta(Vt)

        #Finalmente tengo Y y V para multiplicarlas y obtener W
        W = multiplicar_matrices(Y, V)

    elif (rango(X) == n and n == p):
        # Caso c. Resuelvo W X = Y
        # Como hacer W = Y X⁻¹ es caro, voy a transponer ambos lados de la ec original
        # (W X)^t = Y^t
        # Queda X^t W^t = Y^t

        # Busco la descomposiicon LU de X^t
        L, U, _ = calculaLU(traspuesta(X))

        #Así ahora tengo LU W^t = Y^t

        #Llamo Z a U W^t. Mequeda L Z = Y^t
        #Como L es triangular inferior lo resuelvo con forward_substitution
        Z = res_tri_matricial(L, traspuesta(Y), inferior=True)

        # Ese Z era U W^t. Y como U es triangular superior lo resuelvo con backward_substitution
        Wt = res_tri_matricial(U, Z, inferior=False)

        W = traspuesta(Wt)

    return W

# Ejercicio 3:

def pinvSVD(U, S, V, Y):
    # Calcula W usando la pseudo-inversa obtenida a partir de la SVD
    
    # calculo X⁺
    UT = traspuesta(U)
    S[0] = np.diag(1 / np.diag(S[0])) # es inversible porque es rango completo
    ST = traspuesta(S)

    # Entonces aca tengo X⁺ = V ST UT
    V = traspuesta(V)
    n = U.shape[0]

    #Particiono U V
    U1, U2 = U[:, :n], U[:, n:]
    V1, V2 = V[:, :n], V[:, n:]

    #Calculo el producto VST
    VST = multiplicar_matrices(V1,ST)

    #Finalmente encuentro W = Y X⁺ = Y V ST UT
    W = multiplicar_matrices(Y,multiplicar_matrices(VST,traspuesta(U1)))

    return W


# Ejercicio 4:

def pinvHouseHolder(Q, R, Y):
    # Tengo Q: p × p (2000 x 2000), R: p × n (2000 x 1536), Y: k × p (2 x 2000)
    
    # Defino dimensiones de trabajo
    p = R.shape[0] # p = 2000 (muestras)
    n = R.shape[1] # n = 1536 (features)
    
    # Calculo V = Y * Q
    # V (k x p) = Y (k x p) @ Q (p x p)
    V = multiplicar_matrices(Y, Q) 
    
    # Extraemos R0: La submatriz triangular superior CUADRADA (n x n)
    R0 = R[:n, :n] # R0 es (1536 x 1536)
    
    # Calculo la matriz del sistema: R0^T
    RT_cuadrada = traspuesta(R0) # RT_cuadrada es (1536 x 1536)
    
    # Calculo el lado derecho V^T y TRUNCAR
    VT_completo = traspuesta(V) # (2000 x 2)
    VT_truncado = VT_completo[:n, :] # (1536 x 2)
    
    # Resolvemos R0^T * W^T = V^T_truncado
    # Argumentos: (1536 x 1536) y (1536 x 2) -> Compatible.
    Wt = resolver_sistema(RT_cuadrada, VT_truncado)
    
    # Por ultimo transponemos para W
    W = traspuesta(Wt) # W es (2 x 1536)

    return W

def pinvGramSchmidt(Q, R, Y):
    # Igual idea que Householder pero usando QR obtenido por Gram-Schmidt

    # Q: p x n (200 x 1536)
    # R: n x n (1536 x 1536)
    # Y: k x p (2 x 200)
    
    # Devuelve: W: k x n (2 x 1536)
    
    
    # Calcular V = Y * Q (Paso V = Y Q)
    # V (k x n) = Y (k x p) @ Q (p x n)
    V = multiplicar_matrices(Y, Q) # V es (2 x 1536)
    # V = Y @ Q  # V es (2 x 1536) -> descomentar para que corra más rápido
    
    # Resolver el sistema R^T * W^T = V^T para W^T
    # W^T = (R^T)^{-1} V^T
    
    RT = traspuesta(R) # R^T es (1536 x 1536). Esta es la matriz triangular L
    VT = traspuesta(V) # V^T es (1536 x 2). Este es el lado derecho Q
    
    # W_T = resolver_sistema(L, Q) = resolver_sistema(R^T, V^T)
    # R^T (1536 x 1536) tiene el tamaño correcto para V^T (1536 x 2)
    W_T = resolver_sistema(RT, VT) # W_T es (1536 x 2)
    
    # Transponer W^T para obtener W
    W = traspuesta(W_T) # W es (2 x 1536)

    return W


# Ejercicio 5:

def esPseudoInversa(X,pX, tol=1e-8):
    # Chequea las 4 condiciones de Moore-Penrose para ver si pX es pseudo-inversa de X
    # Para que se cumpla moore-penrose tenemos que validar
    # a) X pX X = X
    cond1 = matricesIguales(multiplicar_matrices(multiplicar_matrices(X,pX), X) , X, tol) 
    
    # b) pX X pX = pX
    cond2 = matricesIguales(multiplicar_matrices(multiplicar_matrices(pX, X), pX),pX, tol)

    # c) hermitiana(X pX). Osea conjugada(A * pseudA) = pseudA*A
    cond3 = matricesIguales(conjugada(multiplicar_matrices(X, pX)), multiplicar_matrices(X, pX),tol) 

    # d) hermitiana(pX X). Osea conjugada(pseudA*A) = pseudA*A
    cond4 = matricesIguales(conjugada(multiplicar_matrices(pX, X)), multiplicar_matrices(pX, X),tol) 
    
    return (cond1 and cond2 and cond3 and cond4)


#Funciones del modulo ALC

#Labo 1
def error(x,y):
    return abs(x - y)

def error_relativo(x,y):
    if x == 0:
        return float('inf')
    
    return abs(x - y) / abs(x)

def matricesIguales(A,B,tol=1e-12):
    #En el labo era sin tol pero para el tp lo agregamos
    if A.shape != B.shape:
        return False
    
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            if abs(A[i, j] - B[i, j]) >= tol:
                return False
    return True


#Labo 2
def rota(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, -s], [s,  c]])


def escala(s):
    n = len(s)
    M = np.zeros((n, n))

    for i in range(n):
        M[i, i] = s[i]

    return M

def rota_y_escala(theta,s):
    if len(s) != 2:
       return None
    R = rota(theta)
    S = escala(s)
    return multiplicar_matrices(S, R)

def afin(theta, s, b):
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
    v_aux = [v[0], v[1], 1.0] # le agregamos un 1 al final
    A = afin(theta, s, b)
    if A is None:
       return None
    w_aux = multiplicar_matrices(A, v_aux)
    w = [w_aux[0], w_aux[1]] 
    return w

#Labo 3
def norma(x, p):
    if p == 1: #norma 1
       return sum(abs(i) for i in x) 
    
    if p == 2: #norma euclidea
       return (sum(i**2 for i in x ))**0.5
    
    if p == 'inf': #norma infinito
        acum = 0
        for r in range(x.shape[0]):
            sum_abs=0
            for c in range(x.shape[1]):
                sum_abs += abs(x.shape[1])
            acum = max(acum, sum_abs)
        return acum
    return None

def normaliza(X, p):
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
    # Aproxima la norma inducida ||A||_{q,p} por Monte Carlo.
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
        y = multiplicar_matrices(A, x)

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
    result = []
    if 1 in p:
       result.append(norma(A, 1))
    if 2 in p:
       result.append(norma(A, 2))
    if 'inf' in p :
       result.append(norma(A, 'inf'))

    return result  

def condMC(A, p):
    normaA, _ = normaMatMC(A, p, p, 10)
    normaInv, _ = normaMatMC(inversa(A), p, p, 10)
    return normaA * normaInv

def condExacto(A,p):
    normA = normaExacta(A, [p])
    normInv = normaExacta(inversa(A), [p])
    return normA[0] * normInv[0]

#Labo 4
def calculaLU(A): 
    #Si no tiene factorización devuelve None
    n, m = A.shape
    
    if m!=n:
        return None
    
    U = A.copy()
    L = np.eye(n)
    cant_op = 0

    for i in range(n):
        pivote = U[i][i]

        if abs(pivote) < 1e-12:
            return None
        
        for j in range(i+1,n):
            if U[j][i] != 0:
                x =  U[j][i] / pivote
                L[j][i] = x
                U[j] = U[j] - (U[i]* x)
                cant_op += 1
     
    return L, U, cant_op

def res_tri(L, b, inferior=True):
  # Resuelve el sistema Lx = b, donde L es triangular. Si inferior es True, L es triangular inferior, si es False, L es triangular superior. '''

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

def inversa(A): #Esta no la usamos, muy cara
    L,U,_ = calculaLU(A)
    n = A.shape[0]
    id = np.eye(n)
    filas = []
    
    for i in range(n):
        y = res_tri(L,id[i],True)
        filas.append(res_tri(U,y,False))
    invertido = traspuesta(np.array(filas))
    
    return invertido

def calculaLDV(A):
  # Calcula la factorizacion LDV de la matriz A, para no usar la inversa lo hacemos así:
  L, U, _ = calculaLU(A)
  Vt, D, _ = calculaLU(traspuesta(U))
  V = traspuesta(Vt)

  return L, D, V

def esSDP(A, atol=1e-8):

    if not matricesIguales(A, traspuesta(A), atol):
        return False

    L, D, V = calculaLDV(A)

    for i in range(D.shape[0]):
        if D[i, i] <= atol:
            return False

    return True
    
def calculaCholesky(A, atol=1e-10):
    A = np.array(A, dtype=float)

    if esSDP(A, atol):
        L, D, V = calculaLDV(A)

        D_sqrt = np.zeros_like(D)
        for i in range(D.shape[0]):
            D_sqrt[i, i] = np.sqrt(D[i, i])

        R  = multiplicar_matrices(L, D_sqrt)
        Rt = multiplicar_matrices(D_sqrt, V)
        
        return R, Rt
    else:
        return None


#Labo 5

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    
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
        R[j,j] = norma(q,2)
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
    
def QR_con_HH(A, tol=1e-12):
    A = A.astype(float, copy=True)   # copiamos para no modificar la original
    m, n = A.shape

    if m < n:
        print("A debe tener m >= n")
        return None

    R = A.copy()
    Qt = np.eye(m)

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
                s += v[i] * Qt[k+i, j]
            coef = 2.0 * s / vTv
            for i in range(len(v)):
                Qt[k+i, j] = Qt[k+i, j] - coef * v[i]

    # Limpiar pequeños errores numéricos debajo de diagonal
    for i in range(1, m):
        # solo iterar hasta el número de columnas (n), porque R es m x n (m >= n)
        for j in range(min(i, n)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0

    Q = traspuesta(Qt)
    return Q, R    # Devolvemos Q ortogonal (transpuesto porque fuimos aplicando desde la izquierda)

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


#Labo 6
def metpot2k(A, tol=1e-15, K=1000):
    """    
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
       w = multiplicar_matrices(A, x.reshape(n,1))
       w = w.reshape(n)
       if norma(w,2) != 0:
          x = w / norma(w, 2)
       if norma( x - x_old, 2) < tol:
          break
       
       it += 1
    # λ≈xkT​Axk​
    Ax = multiplicar_matrices(A, x.reshape(n,1))
    lamb = multiplicar_matrices(x.reshape(1,n), Ax.reshape(n,1))
    return x, lamb[0][0], it

def diagRH(A, tol=1e-15, K=1000):
    """
    Diagonalización usando reflexiones de Householder.
    """
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
       vvT = multiplicar_matrices(v, v.T)
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

    HA = multiplicar_matrices(H, A)
    B = multiplicar_matrices(HA, traspuesta(H))
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
    S_final = multiplicar_matrices(H, S_block)
    return S_final, D_final


#Labo 7
def transiciones_al_azar_continuas(n):
    matriz_aleatoria = np.random.uniform(0.0, 1.0, (n,n))
    for j in range(n):
       suma_col = np.sum(matriz_aleatoria[:, j])
       if suma_col != 0:
          matriz_aleatoria[:, j] = matriz_aleatoria[:, j] / suma_col
       else:
          matriz_aleatoria[:, j] = np.ones(n) / n
    return matriz_aleatoria      

def transiones_al_azar_uniforme(n, thres):
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
    m, n = A.shape
    A = np.array(A, dtype=float)
    AT = traspuesta(A)
    ATA = multiplicar_matrices(AT, A) 
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

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):

    if not (len(listado[0]) == len(listado[1]) == len(listado[2])):
       return None
    diccionario = dict()
    for i in range(len(listado[0])):
       if abs(listado[2][i]) > tol:
          diccionario[(listado[0][i], listado[1][i])] = listado[2][i] 
    return diccionario, (m_filas, n_columnas)
    
def multiplica_rala_vector(A,v):
    diccionario, tupla = A
    w = np.zeros(tupla[0])
    for (i,j), aij in diccionario.items():
       w[i] += aij * v[j] 
    return w 


#Labo 8

def svd_reducida(A, k="max", tol=1e-15):
    A = np.array(A, dtype=float)
    m, n = A.shape

    if m >= n:
       AT = traspuesta(A)
       ATA = multiplicar_matrices(AT, A)
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
           
       print("SIGMAS ALL (m<n):", sigmas_all)
       print("KEEP IDX (m<n):", keep_idx)
       if keep_idx.size == 0:
          return np.zeros((m,0)), np.array([]), np.zeros((n,0))

       V_reducida = V_all[:, keep_idx].copy()

       # asegurar normalizacion de columnas
       for i in range(V_reducida.shape[1]):
           nv = norma(V_reducida[:, i],2)
           if nv > 0:
               V_reducida[:, i] = V_reducida[:, i] / nv

       sigmas = sigmas_all[keep_idx]

       # proteger contra NaN/inf
       if not np.all(np.isfinite(sigmas)):
           sigmas = sigmas[np.isfinite(sigmas)]
           if sigmas.size == 0:
               return np.zeros((m,0)), np.array([]), np.zeros((n,0))
           V_reducida = V_reducida[:, :len(sigmas)]

       AV = np.array(multiplicar_matrices(A, V_reducida))
       U = np.zeros((m, len(sigmas)))
       for i in range(len(sigmas)):
           if sigmas[i] > tol:
               U[:, i] = AV[:, i] / sigmas[i]
           else:
               U[:, i] = 0.0

       # normalizar U por si acaso
       for i in range(U.shape[1]):
           nu = norma(U[:, i],2)
           if nu > 0:
               U[:, i] = U[:, i] / nu
  
       return U, sigmas, V_reducida

    else:
        AAT = multiplicar_matrices(A, traspuesta(A))
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
            nu = norma(U_reducida[:, i],2)
            if nu > 0:
                U_reducida[:, i] = U_reducida[:, i] / nu

        sigmas = sigmas_all[keep_idx]
        if not np.all(np.isfinite(sigmas)):
            sigmas = sigmas[np.isfinite(sigmas)]
            if sigmas.size == 0:
                return np.zeros((m,0)), np.array([]), np.zeros((n,0))
            U_reducida = U_reducida[:, :len(sigmas)]

        AT = traspuesta(A)
        ATU = np.array(multiplicar_matrices(AT, U_reducida))
        V = np.zeros((n, len(sigmas)))
        for i in range(len(sigmas)):
            if sigmas[i] > tol:
                V[:, i] = ATU[:, i] / sigmas[i]
            else:
                V[:, i] = 0.0

        # normalizar V
        for i in range(V.shape[1]):
            nv = norma(V[:, i],2)
            if nv > 0:
                V[:, i] = V[:, i] / nv

        return U_reducida, sigmas, V


#Mas auxiliares usadas para el tp:

def traspuesta(A):
    A = np.array(A)
    n, m = A.shape
    B = np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i, j]
    return B    

def rango(A, tol=1e-12):
    A = np.array(A, dtype=float)
    n, m = A.shape
    R = A.copy()
    res = 0 #Acá cuento los pivotes para triangular y ver el rango
    fila = 0
    col = 0

    while fila < n and col < m:
        # busco pivote en la columna col
        piv = None
        maxval = 0.0
        for i in range(fila, n):
            if abs(R[i, col]) > maxval:
                maxval = abs(R[i, col])
                piv = i

        #si no encontre paso a la sig columna
        if piv is None or maxval <= tol:
            col += 1
            continue

        #intercambio filas para subir la del pivote
        if piv != fila:
            R[[fila, piv]] = R[[piv, fila]]

        #triangula abajo del pivote
        for i in range(fila + 1, n):
            factor = R[i, col] / R[fila, col]
            R[i, col:] -= factor * R[fila, col:]

        #encontre pivote valido , aumento rango
        res += 1
        fila += 1
        col += 1

    return res

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

def vector_householder(x, tol=1e-12):
    # x es un vector (1D numpy array)
    normx = norma(x,2)
    if normx <= tol:
        return None  # no se puede construir reflector útil

    # Elegimos el signo para evitar cancelación numérica
    alpha = sign(x[0]) * normx

    # Vector v = x + alpha e1
    v = x.copy()
    v[0] = v[0] + alpha

    # Si v es casi cero, no sirve
    normv = norma(v,2)
    if normv <= tol:
        return None

    return v  # no normalizamos aquí

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def conjugada(A):
    #Como trabajamos solo con matrices en R => conjugada(traspuesta(A)) = traspuesta(A)
    return traspuesta(A)

def dot(vec1, vec2):
   if len(vec1) != len(vec2):
      print("deben ser igual tamaño para usar dot")
      return None
   result = 0
   for idx in range(len(vec1)):
    result += vec1[idx] * vec2[idx]
   return result

def res_tri_matricial(L,B,inferior=True): 
    # B matriz m x k
    k = B.shape[1]
    
    X = np.zeros_like(B, dtype=float)

    for j in range(k):
        # Tomar columna j de B
        bj = B[:, j]
        # Resolver L x = bj usando res_tri del labo
        xj = res_tri(L, bj, inferior=inferior)
        X[:, j] = xj

    return X


def resolver_sistema(R, Q):
    
    # Resuelve el sistema matricial R @ X = Q, donde R es triangular.
    # R: n x n (Triangular, R^T)
    #Q: n x k (Lado derecho, V^T)
    # Devuelve X: n x k (W^T)
    
    R = np.array(R, dtype=float, copy=True)
    Q = np.array(Q, dtype=float, copy=True)
    
    m, k = Q.shape # m=1536, k=2
    n = R.shape[0] # n=1536

    # Verificación de Shapes (ajustada a la operación R @ X = Q)
    if n != m or n != R.shape[1]:
        raise ValueError(f"Shapes incompatibles para R @ X = Q. R.shape={R.shape}, Q.shape={Q.shape}")

    # Detección de triangularidad (asumimos que la lógica de inferior_flag en tu módulo es correcta)
    tol = 1e-12
    is_upper = np.allclose(R, np.triu(R), atol=tol)
    
    # R es R^T, que es lower triangular (inferior=True)
    inferior_flag = True
    if is_upper:
        inferior_flag = False
        
    # Inicializar la matriz de solución X (que será W^T)
    X = np.zeros((n, k), dtype=float) 

    # Resolver R * x_j = q_j para cada una de las k columnas (j) de Q
    # Iteramos 2 veces (k=2)
    for col_idx in range(k):
        b = Q[:, col_idx] # b es la columna j de V^T (shape 1536,)
        
        # Resolvemos el sistema vectorial R @ x = b
        x = res_tri(R, b, inferior=inferior_flag)
        
        X[:, col_idx] = x

    # X es W_T (1536 x 2)
    return X

def concatenar_filas(A, B):
    largoFilasA = A.shape[0]
    largoFilasB = B.shape[0]
    columnas = A.shape[1]

    C = np.zeros((largoFilasA + largoFilasB, columnas))

    C[:largoFilasA, :] = A
    C[largoFilasA:, :] = B

    return C

def concatenar_columnas(A, B):
    largoColumnasA = A.shape[1]
    largoColumnasB = B.shape[1]
    filas = A.shape[0]

    C = np.zeros((filas, largoColumnasA + largoColumnasB))

    C[:, :largoColumnasA] = A
    C[:, largoColumnasA:] = B

    return C


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

        R[j, j] = norma(v,2)
        if R[j, j] < tol:
            return None

        Q[:, j] = v / R[j, j]

    return Q, R


