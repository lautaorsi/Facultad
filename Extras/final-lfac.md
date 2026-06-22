
- Ejercicio 10
    Primero vemos que tenemos un APD P y un AP S, sabemos que L(P) = LLCD y L(S) = LLC. 

    - a: 
        -   Verdadero, simplemente tomamos un nuevo q0 con transiciones lambda que conectan los q0 de P y S

    - b:
        -   Falso, los lenguajes libres de contexto no estan cerrados por interseccion.<br> Contraejemplo P = {a^n b^n c^m}, S = {a^n b^m c^m} <br> PnS = {a^n b^n c^n}

    - c:
        -   Falso, los lenguajes LLCD son un subconjunto de lenguajes, por ejemplo {ww^r} es LLC pero no LLCD

    - d:
        -   Verdadero, podemos minimizar los AF y comparar isomorfismo

-   Ejercicio 11
    Invertimos el automata y determinizamos

-   Ejercicio 12
    Sabemos que los LLCD son cerrados por interseccion con lenguajes regulares, gracias a esto podemos calcular la interseccion de ambos lenguajes y obtenemos un LLCD. Una vez hecho esto podemos observar la gramatica correspondiente al automata de pila deterministico resultante de la interseccion y ver que no contenga transiciones del estilo Vn -> (Vt)* Vn (Vt)* | (Vt)* Vn | Vn (Vt)*

-   Ejercicio 13
    -   a:
        -   Falso, si fuera verdadero tendriamos que not(L1 U L2) es reconocible, como el lenguaje es cerrado por union L1UL2 es reconocible por un contador, pero el complemento es not(L1) n not(L2) y ya vimos que la interseccion de lenguajes no es cerrada y por lo tanto el complemento tampoco puede serlo.  

    -   b:
        -   Verdadero, unimos un q0 nuevo a los q0 viejos con transiciones lambda
    -   c:
        -   Falso, un contraejemplo para refutarlo seria tomar L1 y L2 contadores tal que L1nL2 no sea contador, sean L1 = {a^n b^n c^k: k,n >= 0}, L2 = {a^n b^k c^k : k,n >= 0}, luego L1nL2 = {a^n b^n c^n} pero esto no es posible

    -   d:
        -   Verdadero, ignoramos el contador
    -   e
        -   Verdadero, {a^n b^n: n >= 0}

-   Ejercicio 14
    -   Tomemos un lenguaje no reconocible por automatas de pila, por ejemplo {a^n b^n c^n}, es sencillo demostrar que un automata de cola reconoce a este lenguaje si tomamos 

