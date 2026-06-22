# Resumen Final

---

## Conjuntos

### Computable Enumerable

Decimos que un conjunto C es c.e. si corresponde al:

-    Dominio de una [funcion parcialmente computable](#parcialmente-computables)
-    Imagen de una [funcion totalmente computable](#totalmente-computables)

> Todo conjunto c.e. infinito contiene un conjunto infinito computable

---

### Computable

Un conjunto C es computable si existe una funcion *f*: $\mathbb{N}$ -> {0,1} total computable que "decide" si un numero existe en C, mas formalmente: <br>

> $\exists$ *f*: $\mathbb{N}$ -> {0,1}, $\forall$ x . f(x) = 1 <=> x $\in$ C

---
---

## Funciones

### Parcialmente computables

Funciones que pueden estar indefinidas para algunos valores de entrada, ademas son aquellas que se pueden definir a partir de las funciones iniciales por un nuermo finito de aplicaciones de:

-   Composicion
-   Recursion
-   Minimizacion no acotada

---

### Totalmente computables

Funciones que deben estar definidas para todas las posibles entradas, ademas son aquellas que se pueden definir a partir de las funciones inciales or un numero finito de aplicaciones de:

-   Composicion
-   Recursion
-   Minimizacion propia

---
---



## Lenguajes

**Definicion**: <br>
Los lenguajes son conjuntos de palabras que utilizan simbolos de un alfabeto

---

### Lenguajes Regulares

Cualquier lenguaje finito es regular, además los lenguajes regulares son aquellos que se pueden denotar con [Expresiones Regulares](#expresiones-regulares).

Cerrado por:

-   Union (finita)
-   Interseccion (finita)
-   Complemento
-   Concatenacion
-   Reverso
-   Kleene

Estos lenguajes son aceptados por [Automatas Finitos](#automatas-finitos), [Automatas de Pila](#automatas-de-pila) y [Maquinas de Turing](#maquinas-de-turing)

---

### Lenguajes Libres de Contexto

No encuentro definición "casual", pero son los lenguajes que requieren cierto grado de memoria (en particular algo que se pueda hacer en una unica cinta de trabajo), se denotan con [Gramaticas Tipo 2](#tipo-2)

Cerrado por:

-   Union
-   Concatenacion
-   Reversa
-   Kleene

No estan cerrados por:

-   Interseccion    (EXCEPTO con [Lenguajes Regulares](#lenguajes-regulares) LLC ∩ LR = LLC)
-   Diferencia
-   Complemento

Estos lenguajes son aceptados por [Automatas de Pila](#automatas-de-pila) y [Maquinas de Turing](#maquinas-de-turing)

---

### Lenguajes Recursivo Enumerables (c.e.)

Son los lenguajes dados por los conjuntos [Computables Enumerables](#computable-enumerable)

Cerrado por:

-   Union
-   Interseccion

Estos lenguajes son aceptados por [Maquinas de Turing](#maquinas-de-turing)

---
---

## Automatas

**Definicion**: <br>
Son funciones tales que para cualquier palabra te dice si pertenece o no al lenguaje codificado.

---

### Automatas Finitos

Los automatas finitos son aquellos que permiten decidir si palabras pertenecen a [Lenguajes Regulares](#lenguajes-regulares), existen dos tipos a fines practicos:

-   AF  (para un estado *q*, simbolo *a* pueden haber varios estados destino || hay transiciones $\lambda$ )
-   AFD

Es importante notar que el poder expresivo de ambos es identico, o en otras palabras: <br>

> Existe un AFD M tal que L(M) = L <=> Existe un AF N tal que L(N) = L

---

### Automatas De Pila

Los automatas de pila permiten decidir si palabras pertenecen a [Lenguajes Libres de Contexto](#lenguajes-libres-de-contexto), existen dos tipos y **ambos tienen transiciones lambda**:

-   AP  (para un estado *q*, simbolo *a*, pila *b* pueden haber varios estados destino)
-   APD

Es importante notar que el poder expresivo **NO** es identico: <br>

> Hay [Lenguajes Libres de Contexto](#lenguajes-libres-de-contexto) que NO son aceptados por ningun automata de pila deterministico (APD)

---

### Maquinas De Turing

Sirven para distintas cosas:

-   Aceptar elementos en conjuntos c.e. **UNICAMENTE ACEPTAR, NO PUEDE NEGAR** *Aca basicamente responde 1,0*
-   Computar funciones parcialmente computables (Solo si el c.e. denota [Dominio](#computable-enumerable) de una [funcion parcialmente computable](#parcialmente-computables)) *Aca basicamente responde los x que no se cuelgan*
-   Enumerar conjuntos c.e. (Si el c.e. denota [Imagen](#computable-enumerable) de una [funcion totalmente computable](#totalmente-computables))    *Aca basicamente responde los *

---
---

## Expresiones Regulares

**Definicion**: <br>
Es la forma que usamos para denotar un [Lenguaje Regular](#lenguajes-regulares), usando simbolos como `(r|s, rs, r*, r+)`. <br> <br>

Tienen una equivalencia 1 a 1 con [Automatas Finitos](#automatas-finitos), en otras palabras: <br>

> Existe una expresion regular R que denota un lenguaje regular L <=> Existe AF M que acepta L

---
---

## Gramaticas

**Definicion**: <br>
Es la forma que usamos para denotar distintos tipos de lenguajes, existen varias gramaticas, entre ellas:

-   Tipo 3 => [Lenguajes Regulares](#lenguajes-regulares)
-   [Tipo 2](#tipo-2) => [Lenguajes Libres de Contexto](#lenguajes-libres-de-contexto)

---

### Tipo 2

Para cada gramatica libre de contexto hay un automata de pila que acepta el mismo lenguaje, en otras palabras: <br>

> Existe una gramatica T2 G que denota un lenguaje *LLC* L <=> Existe AP M que acepta L