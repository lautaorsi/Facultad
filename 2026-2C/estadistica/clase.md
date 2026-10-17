Variables Aleatorias (1.4.1)
---
Una variable aleatoria es una funcion del espacio muestral a los reales $\R$

Ejemplos

    | Experimentos | Var Aleatoria |
    |   Arrojar dos dados   | X = Suma de los dados  |
    |   Tirar una moneda 25 veces   | X = numero de caras |
    |   Aplicar fertilizante en diferentes smedidas a un cultivo    | X = rendimiento en ton/ha    |
        
Definimos un nuevo espacio muestral para la var aleatoria

Supongamos $S = \{S_1, ..., S_n\}$, una función de probabilidad $P$ y una var aleatoria $X = \{X_1,...,X_n\}$, definimos entonces la funcion de prob $P_x$ de la siguiente manera:

Observamos el valor $X = x_i \iff $ el resultado del exp. es $S_i \in S$ tal que $X(S_j) = X_i$

Probabilidad Inducida

Si X es numerable

$P_X(X=x_i) = P(\{S_j \in S : X(S_j) = X_i\})$

Si X es inumerable

$P_X(X \in A) = P(\{ s \in S : X(s) \in A \})$

> _Notación_  <br> Son equivalentes $P_X(X=x_i) = P_X(x_i) = P(X=x_i)$

ej 1.4.4

Supongamos el espacio S de tamaño 2⁵⁰ de una encuesta con 2 opciones hecha a 50 personas, $S=\{(1,0,...,0), (0,1,...,0), (1,1,...,1)\}$ con resultados equiprobables.

$P_X(X=27) =  \frac{\binom{50}{27}}{2⁵⁰}$


función de distribución acumulada  (1.5.1)
---

X variable aleatoria, su fda $F_X$ es 

$F_X(x) = P_X(X \leq x)$

Ejemplo 1.5.2

Tiramos tres monedas equilibradas, definimos X = número de caras

fda:


F_X(x) = 0 - inf < x < 0

$1/8 \iff 0 <= x < 1$ <br>
$1/2 \iff 1 <= x < 2$ <br>
$7/8 \iff 2 <= x < 3$ <br>
$1 \iff 3< x < inf$

Propiedades de la fda (Teo 1.5.3)
---

Una funcion F es una fda si y solo si

1.  $\lim_{x \to -\infty} F(x)= 0 \land \lim_{x \to +\infty} F(x)= 1$
2.  F es no decreciente
3. F es continua a la derecha


Ejemplo 1.5.4

Arrojamos una moenda hasta que salga una cara

Sea 

p = probabildiad de cara en cualquier tirada <br>
x = numero de tiradas necesarias para obtener una cara

$P_X(X = 1) = p$ <br>

$P_X(X = 2) = (1 - p) * p$

$P_X(X=3) = (1 - p)² * p$

...

$P_X(X=x) = (1-p)^{x-1} * p$

fda

$F_X(X=x) = P_X(X \leq x) = \sum^{x}_{i=1}{P_X(X=i)} = \sum^{x}_{i=1}{(1-p)^{i-1}p}$

Usemos

$\sum^{n}_{k=1}{t^{k-1}} = \frac{1-t^{n}}{1-t}$

$F_X(x) = \frac{1-(1-p)^{x}}{1-(1-p)}*p = 1 - (1 - p)^{x}$ 



Variables aleatorias discretas vs continuas
---

-   Una variable aleatoria X es **continua** si $F_X(x)$ es una función continua de X

-   Una variable aleatoria X es **discreta** si $F_X(x)$ es una función con escaloenes



Variables identicamente distribudias (1.5.8)
---

$x$ e $y$ son identicamente distribuidas si para cada $A \in \Beta$ se verifica que $P(x \in A) = P(y \in A)$

ejemplo 1.5.9

Experimento arrojar una moneda equilibrada tres veces. Definimos 

X = numero de caras
y = numero de secas



Funcion de probabilidad puntual (1.6.1)
---

La fpp de una variable aleatoria discreta X es

 $f_X(x) = P(X=x)$

Relacion con la fda

$F_X(x) = P_X(X \leq x=) = \sum^{x}_{i=!}{P_X(X=i)} = \sum^{x}_{i=!}{f_X(X=i)} $

Funcion de densidad de probabilidad (1.6.3)
---

La fdp de una variable continua X es $f_X(x)$ tal que

$F_X(x) = \int^{x}_{-\infty}{f_X(t) \, dt}$

> Notación: "la variable aleatoria $X$ tiene funcion de distribucion de probabilidad $F_X(x)$"


> Para variables continuas, la probabilidad de obtener un nro exacto es 0 y la probabilidad de obtener un nro en un rango es indiferente de si es cerrado o no.

Probabilidad como area bajo la curva (Ej 1.6.4)
---
$F_X(x) = \frac{1}{1+e^{-x}}$


$P(a<x<b) = F_X(b) - F_X(a) = \int^{b}_{-\infty} f_X(t) \, dt - \int^{a}_{-\infty} f_X(t) \, dt = \int^{b}_{a} f_X(t) \, dt$  

Condicion para fdp o fpp (1.6.5)
---

Una funcion f(x) es fdp/fpp de X variable aleatoria si y solo si

a.  $\forall x.  f_X(x) \geq 0$

b.  $\begin{cases} 
\int^{-\infty}_{-\infty} f_X(x) \, dx = 1 & \text{(fdp)} \\[10pt] 
\sum_x{f_X(x)} & \text{(fpp)} 
\end{cases}$



Valor esperado
---
El valor esperado de una variable aleatoria X, llamado E(x) es 

$E(X) = \begin{cases} 
\int^{\infty}_{-\infty} x f_X(x) \, dx & \text{si } X \text{ es continua} \\[10pt] 
\sum_x x f_X(x) & \text{si } X \text{ es discreta} 
\end{cases}$


En general, el valor esperado de g(x)

$E[g(x)] = \begin{cases}
 \int^{\infty}_{-\infty} f_X(x) \, dx  & \text{si } X \text{ continua}\\[10pt]
\sum_x g(x) f_X(x) & \text{si } X \text{ discreta}
\end{cases}$

Dos valores esperados muy habituales:

$g(x) = x \rarr E(x) = \text{promedio (ponderado)}$

$g(x) = (x- E(x))² \rarr E[(x-E(x))²] = \text{varianza}$

$\sqrt{\text{varianza}} = \text{std (tiene las mismas unidades que x y E(x))}$

Propiedades de valor esperado
---

X variable aleatoria, a,b,c constances, $g_1(x),g_2(x)$ funciones con valor esperado finito

1.  $E[ag_1(x) + b_2(x) + c] = aE[g_1(x)] + bE[g_2(x)] + c$

2.  Si $\forall x. \quad g_1(x) \geq 0 \rarr E[g_i(x)] \geq 0$ 

3.  Si $\forall x. \quad g_1(x) \geq g_2(x) \rarr E[g_1(x)] \geq E[g_2(x)]$

4.  Si $\forall x. \quad a \leq g_1(x) \leq b \rarr a \leq E[g_1(x)] \leq b$