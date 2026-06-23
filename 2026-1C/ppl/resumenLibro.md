[Link al Libro](https://arxiv.org/pdf/1809.10756)





clase 02: 2.1, 2.2.1, 2.2.2, 4.1 

clase 03: 4.2, 4.2.1, 3.2





# Capitulo 1: Introducción

## 1.1: Razonamiento basado en Modelos

Un modelo es una construccion artificial diseñada para responder de forma idéntica al sistema que se quiere entender.

Los modelos numéricos y simulaciones por computadora reemplazaron los modelos físicos y son por naturaleza aproximaciónes. Dichos modelos numéricos emulan la estocacidád utilizando generadores pseudoaleatorios de numeros para simular fenómenos _random_.

Llammos **observaciones** a aquellos valores producidos por los modelos y que son medibles en el mundo real. 

### 1.1.1: Denotación de Modelos

Un ejemplo de modelo estadístico es el modelo beta-Bernoulli para simular el giro de una moneda, típicamente denotado como

$$x \sim \text{Beta}(\alpha, \beta)$$

$$y \sim \text{Bernoulli}(x)$$

En este contexto α y β son parámetros, x es una variable latente e y es el valor resultante de girar la moneda.

> Una variable latente es aquella que se elije de un rango de valores con algún típo de distribución, algo así como una variable "al azar".


### Regla de Bayes

Esta regla nos indica cómo derivar una probabilidad condicional, condicionar nos indica cómo actualizar nuestras creencias.

$$p(X|Y) = \frac{p(Y|X)p(X)}{p(Y)} = \frac{p(X, Y)}{p(Y)} = \frac{p(X, Y)}{\int p(X, Y) \, dX}$$

Alguas aclaraciones:

-   $ p(X|Y) $ es el _likelihood_ (o probabilidad de verosimilitud) <br> _**La probabilidad de que ocurra X sabiendo que ocurrió Y**_
-   $ p(X) $ es el _prior_ (o probabilidad a priori) <br>
    _**La probabilidad de que ocurra X antes de analizar Y**_
-   $ P(Y)$ es el _marginal likelihood_ (o verosimilitud marginal) <br>

-   $ P(Y|X)$ es el _posterior_ (o probabilidad a posteriori) <br>
    _**La probabilidad de que ocurra X despues de analizar Y**_
-   $ P(X,Y)$ es el _joint likelihood_ (o verosiomilitud conjunta)
    _**La probabilidad de que ocurran X e Y a la vez**_
### Condicionamiento

Condicionar es parametrizar una distribución, recordemos que tenemos **P(X|Y)** _(o la probabilidad de que, dado Y, ocurra X)_, condicionar sería el acto de fijar el valor de Y para obtener una distribución específica de X según el valor fijado de Y.

En el contexto del giro de moneda teníamos la **variable latente X** que representaba el **_bias_** de la moneda (la probabilidad de que la moneda caiga en cara o seca) y la **variable Y** que representa el **valor resultante del giro**.

A partir de eso aplicar el condicionamiento P(X|Y = _heads_) nos permite actualizar el _bias_ de la moneda, dado que cayó en cara la curva de probabilidad cara/seca debe inclinarse levemente más a cara. 

## 1.2: Programación Probabilística

## 1.4: Primer Programa Probabilistico

## 1.5: Primer Evaluador de Programa Probabilístico





# Capitulo 2: Lenguaje Probabilístico Sin Recursión

Lenguaje de Programación Probabilistica de Primer Orden (FOPPL)

## 2.1: Syntax

### General

| Regla | Descripción |
| :--- | :--- |
| **$v$** | Variable _(references value of another expression in the program)_ |
| **$c$** | Constant value _(number, str, bool, vector...)_ or primitive operation _(+...)_ |
| **$f$** | Procedure |
| **$e $** | $c$ \| $v$ \| <br> `(let [v e1] e2)` _(assigns **e1** to **v**, can be accessed in **e2**)_ <br> \| `(if e1 e2 e3)` _(if (**e1**) **e2** else **e3**)_ <br> \| `(f e1 ... en)` _(f(**e1**,...,**en**))_ <br> \| `(c e1 ... en)` _(same as f, but c is primitive function)_ <br> \| `(sample e)` _(returns a sample value from e, which has to be a distribution object)_ <br>\| `(observe e1 e2)` _(e1 has to be a distribution, e2 is the actual value used for conditioning)_|
| **$q $** | $e$ \| `(defn f [v1 ... vn] e) q` |

> Un programa _q_ puede ser una única expresión _e_ o una función <br> _(defn f ...)_ seguida por cualquier programa _q_.

### Vectores
-   `(first e)` First element of a list/vector ***e***
-   `(rest e)` Tail of a list/vector ***e***
-   `(last e)` last element of list/vector ***e***
-   `(append e1 e2)` appends ***e2*** to the end of ***e1***
-   `(get e1 e2)` retrieves element at index/key ***e2*** from list/vector/hashmap ***e1***
-   `(put e1 e2 e3)` replaces element at index/key ***e2*** with value ***e3*** in vector/hashmap ***e1***
-   `(remove e1 e2)` removes element at index/key ***e2*** in vector/hashmap ***e1***

>! IMPORTANTE: Structure modifications do NOT happen in-place but rather are RETURNED as MODIFIED COPIES

### Ejemplo

Veamos un ejemplo de regresión lineal. En líneas generales lo que busca este código es definir una distribucion de lineas expresadas en términos de sus pendientes y ordenadas al origen, esto se logra armando una distribución a _priori_ y despues se usan 5 datapoints para condicionar.

```clojure
(defn observe-data [slope intercept x y] ;
  (let [fx (+ (* slope x) intercept)]
    (observe (normal fx 1.0) y)))

(let [slope (sample (normal 0.0 10.0))]
  (let [intercept (sample (normal 0.0 10.0))]
    (let [y1 (observe-data slope intercept 1.0 2.1)]
      (let [y2 (observe-data slope intercept 2.0 3.9)]
        (let [y3 (observe-data slope intercept 3.0 5.3)]
          (let [y4 (observe-data slope intercept 4.0 7.7)]
            (let [y5 (observe-data slope intercept 5.0 10.2)]
              [slope intercept])))))))
```

- `observe-data` es una función que toma un par (x,y) y condiciona al modelo observando el valor de y.

- El nido de `let` define una pendiente y _oo_ al azar de una distribución normal, despues se condiciona con los 5 datos para tener una distribución posterior en base a los mismos.

### 2.2.1: Let forms

Como se ve en el ejemplo anterior, las funciones `Let` son bastante engorrosas ya que solo permiten declarar una variable por llamado, es por esto que se puede generalizar usando 

```clojure
    (let [v1 e1
        ;...
        ;...
        vn en
    ]
    en+1 ... em)
```

Por otro lado tambien esta bueno notar que la función observe(e1 e2) retorna siempre e2 despues de condicionar e1, entonces originalmente teniamos un monton de variables yn que no sirven para nada, podemos aplicar algo similar a haskell/python y hacer 

```clojure
    (let [ _ ( observe ( normal 0 1) 2.0)] . . .)
```

### 2.2.2: For loops

En simultaneo podemos ver del ejemplo que repetir la función (observe-data ...) es engorroso y por lo tanto usar un for loop nos ayuda bastante

```clojure
    (foreach c
    [v1 e1 ... vn en]
    e1 ... ek)
```

Entonces reescribiendo el ejemplo optimizado con for loops y Let

```clojure
    (let [  y-values [2.1 3.9 5.3 7.7 10.2]
            slope ( sample ( normal 0.0 10.0))
            intercept ( sample ( normal 0.0 10.0))]
        (foreach 5
            [ x (range 1 6)
              y y-values]
            (let [ fx (+ (* slope x ) intercept )]
                ( observe ( normal fx 1.0) y )))
        [slope intercept])
```



# Captiulo 3

## 3.2: Evaluating Density


# Capitulo 4

## 4.1: Likelihood Weighting

Es uno de los metodos de inferencia basados en evaluación más simples. 

### 4.1.1: Importance Sampling

Importance sampling aproxima la distribución posterior p(X|Y) usando samples. El truco es cambiar el saple 

Que operaciones tienen que hacerse

Como implementar likelihood weighting

Como implementar importance sampling corriendo un programa muchas veces


## 4.2: Metropolis-Hastings

Son metodos que generan una cadena de Markov de los valores retornados por programas $\text{r(X)}^{1}\text{,...,r(X)}^{\text{S}}$ aceptando o rechazando samples según un algoritmo

  1. Inicializa un resultado r con una "calidad" W
  2. En cada paso genera un nuevo resultado r' con una calidad W'
  3. Compara las calidades y si el nuevo es mejor lo adopta, si es peor le da una oportunidad según que tan malo sea
  4. Si el nuevo resultado es rechazado se queda con el anterior  


### 4.2.1: Single-Site Proposals

Es la idea de no recalcular todas las variables aleatorias para incrementar la probabilidad de aceptación de un nuevo resultado.

  1. Se define una única variable a recalcular
  2. Copia el resto de variables de la ejecución anterior
  3. Corre el programa usando las copias y recalculando variable elegida
  4. Resuelve y decide si se queda con el cambio o descarta


