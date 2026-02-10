import numpy as np




def rota(theta):
    matrizRotacion = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])

    return matrizRotacion

def escala(s):
    matriz = np.zeros([len(s), len(s)])
    for i in range(len(s)):
        matriz[i][i] = s[i]
    return matriz

def rotayescala(theta, s):
    matriz =  np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])

    for i in [0,1]:
        matriz[i][i] *= s[i]
    
    return matriz

def afin(theta,s,b):
    matriz =  np.zeros([3,3])
    matriz[0][0] = np.cos(theta)
    matriz[0][1] = -np.sin(theta)
    matriz[1][0] = np.sin(theta)
    matriz[1][1] = np.cos(theta)

    for i in range(len(s)):
        matriz[i][i] *= s[i]
    
    for i in range(len(b)):
        matriz[i][2] = b[i]

    matriz[2][2] = 1
    return matriz

def transafin(v,theta,s,b):
    print(afin(theta,s,b))
    return productoMatricial(v,afin(theta,s,b))

def main():
    # Tests para rota
    assert(np.allclose(rota(0), np.eye(2)))
    assert(np.allclose(rota(np.pi / 2), np.array([[0, -1], [1, 0]])))
    assert(np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]])))

    # Tests para escala
    assert(np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]])))
    assert(np.allclose(escala([1, 1, 1]), np.eye(3)))
    assert(np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])))

    # Tests para rota y escala
    assert(np.allclose(rotayescala(0, [2, 3]), np.array([[2, 0], [0, 3]])))
    assert(np.allclose(rotayescala(np.pi / 2, [1, 1]), np.array([[0, -1], [1, 0]])))
    assert(np.allclose(rotayescala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]])))

    # Tests para afin
    assert(np.allclose(
        afin(0, [1, 1], [1, 2]),
        np.array([
            [1, 0, 1],
            [0, 1, 2],
            [0, 0, 1]
        ])
    ))

    assert(np.allclose(
        afin(np.pi / 2, [1, 1], [0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    ))

    assert(np.allclose(
        afin(0, [2, 3], [1, 1]),
        np.array([
            [2, 0, 1],
            [0, 3, 1],
            [0, 0, 1]
        ])
    ))

    # Tests para transafin
    assert(np.allclose(
        transafin(np.array([1, 0]), np.pi / 2, [1, 1], [0, 0]),
        np.array([0, 1])
    ))

    assert(np.allclose(
        transafin(np.array([1, 1]), 0, [2, 3], [0, 0]),
        np.array([2, 3])
    ))

    assert(np.allclose(
        transafin(np.array([1, 0]), np.pi / 2, [3, 2], [4, 5]),
        np.array([0, 7])
    ))

if __name__ == "__main__":
    main()




