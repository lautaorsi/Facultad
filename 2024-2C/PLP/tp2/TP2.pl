%%%%%%%%%%%%%%%%%%%%%%%%
%% Predicados básicos %%
%%%%%%%%%%%%%%%%%%%%%%%%

%% Ejercicio 1
%% proceso(+P)
proceso(computar).
proceso(escribir(_,_)).
proceso(leer(_)).
proceso(secuencia(X,Y)) :-  proceso(X), proceso(Y).
proceso(paralelo(X,Y)) :- proceso(X), proceso(Y).

%% Ejercicio 2
%% buffersUsados(+P,-BS)
buffersUsados(computar,[]).
buffersUsados(escribir(B,_),[B]).
buffersUsados(leer(B),[B]).
buffersUsados(secuencia(P,Q),BS) :- buffersUsados(P,YS), buffersUsados(Q,XS), append(YS,XS,ZS), sort(ZS,BS).
buffersUsados(paralelo(P,Q),BS) :- buffersUsados(P,YS), buffersUsados(Q,XS), append(YS,XS,ZS), sort(ZS,BS).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Organización de procesos %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Ejercicio 3
%% intercalar(+XS,+YS,?ZS)
intercalar([],[],[]).
intercalar([X|XS],YS, [X|ZS]) :- intercalar(XS, YS,ZS).
intercalar(XS, [Y|YS], [Y|ZS]) :- intercalar(XS,YS,ZS).

%% Ejercicio 4
%% serializar(+P,?XS)
serializar(computar,[computar]).
serializar(leer(B),[leer(B)]).
serializar(escribir(B,E),[escribir(B,E)]).
serializar(secuencia(P,Q),XS) :- serializar(P,PS), serializar(Q,QS), append(PS,QS,XS).
serializar(paralelo(P,Q),XS) :- serializar(P,PS), serializar(Q,QS), intercalar(PS,QS,XS).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contenido de los buffers %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Ejercicio 5
%% contenidoBuffer(+B,+ProcesoOLista,?Contenidos)
contenidoBuffer(_,[],[]).
contenidoBuffer(_,computar,[]).
contenidoBuffer(B,escribir(B,_),[B]).
contenidoBuffer(B,leer(NOTB),[]) :- NOTB \= B.
contenidoBuffer(B,escribir(NOTB,_),[]) :- NOTB \= B.
contenidoBuffer(B,Z,C) :- proceso(Z), Z \= computar,Z \= leer(_), Z \= escribir(_,_), serializar(Z, XS),esSerieSegura(false,B,XS),contenidoBuffer(B,XS,C).
contenidoBuffer(B,[X|XS],C) :- esSerieSegura(false,B,[X|XS]), contenidoDeBufferEnLista(B,[X|XS],C1),cantLecturas(B,[X|XS], N), sacarPrimeros(N,C1,C).

esSerieSegura(_,_,[]).
esSerieSegura(Z,B,[computar|XS]) :- esSerieSegura(Z,B,XS).
esSerieSegura(true,B,[leer(B)|XS]) :- esSerieSegura(false,B,XS).
esSerieSegura(Z,B,[leer(NOTB)|XS]) :- NOTB \= B, esSerieSegura(Z,B,XS).
esSerieSegura(_,B,[escribir(B,_)|XS]) :- esSerieSegura(true,B,XS).
esSerieSegura(Z,B,[escribir(NOTB,_)|XS]) :- NOTB \= B,  esSerieSegura(Z,B,XS).

contenidoDeBufferEnLista(_,[],[]).
contenidoDeBufferEnLista(B,[escribir(B,X)|XS],[X|C]) :- contenidoDeBufferEnLista(B,XS,C).
contenidoDeBufferEnLista(B,[Y|XS],C) :- Y \= escribir(B,_), contenidoDeBufferEnLista(B,XS,C).

cantLecturas(_,[],0).
cantLecturas(B,[leer(B)|XS], N) :- cantLecturas(B,XS,N1), N is N1+1.
cantLecturas(B,[Y|XS], N) :- Y \= leer(B),cantLecturas(B,XS,N).

sacarPrimeros(0,XS,XS).
sacarPrimeros(N,[_|XS],C) :- N \= 0, N1 is N-1 ,sacarPrimeros(N1,XS,C). 




%% contenidoLeido(+ProcesoOLista,?Contenidos)
contenidoLeido(X,[]) :- proceso(X), X \= leer(), X \= secuencia(_,_),X \= paralelo(_,_).
contenidoLeido(Z,C):- proceso(Z), Z \= leer(_),Z \= escribir(_,_),Z \= computar,serializar(Z,XS), contenidoLeido(XS,C).
contenidoLeido(XS,C):- separarUltimo(XS,U,YS), recorroDesdeAtras(U,YS,C). %%recibo siempre una lista

%%recorroDesdeAtras(+UltimoElem,+Lista,?Contenidos)
recorroDesdeAtras(X,[],[]):- X \= leer(_).
recorroDesdeAtras(leer(B),XS,C):- contenidoBuffer(B,XS,[E|_]), separarUltimo(XS,U,YS), recorroDesdeAtras(U,YS,C1), append(C1,[E],C).
recorroDesdeAtras(U,XS,C):- U\=leer(_), separarUltimo(XS,X,YS), recorroDesdeAtras(X,YS,C), !.

separarUltimo([U],U,[]).
separarUltimo(XS,U,YS):- append(YS,[U],XS).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contenido de los buffers %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Ejercicio 7
%% esSeguro(+P)
esSeguro(computar).
esSeguro(escribir(_,_)).
esSeguro(secuencia(P,Q)) :-verificarParalelos(P),verificarParalelos(Q),findall(ZS,serializar(secuencia(P,Q),ZS),RS),forall(member(Z,RS),contenidoLeido(Z,_)).
esSeguro(paralelo(P,Q)):-verificarParalelos(P),verificarParalelos(Q), buffersUsados(P,XS),buffersUsados(Q,YS), noCompartenBuffer(XS,YS), findall(ZS,serializar(paralelo(P,Q),ZS),RS),forall(member(Z,RS),contenidoLeido(Z,_)).

verificarParalelos(leer(_)).
verificarParalelos(computar).
verificarParalelos(escribir(_,_)).
verificarParalelos(paralelo(P,Q)):- buffersUsados(P,XS),buffersUsados(Q,YS),noCompartenBuffer(XS,YS),verificarParalelos(P),verificarParalelos(Q).
verificarParalelos(secuencia(P,Q)):-verificarParalelos(P),verificarParalelos(Q).

noCompartenBuffer([],_).
noCompartenBuffer([X|XS],YS):- not(member(X,YS)), noCompartenBuffer(XS,YS).

%% Ejercicio 8
%% ejecucionSegura(?XS,+BS,+CS) - COMPLETAR LA INSTANCIACIÓN DE XS


%%(En length(XS,N) le estamos asignando N para poder referenciarla en la justificacion, se podria usar length(XS,_))
ejecucionSegura(XS, BS, CS) :- length(XS,N), generarListaDeProcesos(XS,BS,CS), listaSegura(XS).


generarListaDeProcesos([],_,_).
generarListaDeProcesos([computar|XS], BS, CS) :-generarListaDeProcesos(XS, BS, CS).
generarListaDeProcesos([escribir(A,B)|XS], BS, CS) :-member(A,BS),member(B,CS),generarListaDeProcesos(XS,BS,CS).
generarListaDeProcesos([leer(A)|XS],BS,CS) :-member(A,BS),generarListaDeProcesos(XS,BS,CS).

listaSegura([]).
listaSegura(XS):- contenidoLeido(XS,_).

  %% 8.1. Analizar la reversibilidad de XS, justificando adecuadamente por qué el predicado se comporta como
  %% lo hace.
%%el predicado es reversible en XS. Podemos ver 3 casos.
%%Si XS no está instanciada: se generan listas de longitud creciente empezando desde 0 mediante length(XS,N). Luego generarListaDeProcesos da, mediante member, todas las posibles listas de acciones básicas de longitud N cuyos buffers (en escribir y leer) estén en BS y contenidos (en escribir) estén en CS. Luego verifica en cada lista generada que no falle en la/s lectura/s, generando asi listas de acciones básicas segura con buffers de BS y contenidos de CS.
%%Si XS está instanciada: length va a instanciar en N el tamaño de XS, generarListaDeProcesos va a verificar mediante member que, por cada proceso, los buffers y contenidos usados en XS pertenecen a BS y CS respectivamente. Luego, listaSegura verifica que en XS no falle la lectura.
%%Si XS está parcialmente instanciada (si hay alguna variable dentro de la lista, ejemplo [X,escribir(1,a),leer(1)] o [computar,escribir(X,a),leer(2)]): length igual que en el caso anterior, va a instanciar en N la longitud de la lista. generarListaDeProcesos va a instanciar las variables, según corresponda, en acciones básicas posibles(computar, escribir o leer con buffers de BS y contenidos de CS) o en buffers/contenidos que aparezcan en BS/CS(mediante member). Solo se van a devolver las instanciaciones que hagan que la lista no falle en la lectura.



%%%%%%%%%%%
%% TESTS %%
%%%%%%%%%%%




cantidadTestsBasicos(14). % Actualizar con la cantidad de tests que entreguen
testBasico(1) :- proceso(computar).
testBasico(2) :- proceso(secuencia(escribir(1,pepe),escribir(2,pipo))).
testBasico(3) :- proceso(escribir(8,hola)).
testBasico(4) :- proceso(leer(8)).
testBasico(5) :- proceso(paralelo(leer(2),leer(1))).
testBasico(6) :- buffersUsados(escribir(1, hola), [1]).
testBasico(7) :- buffersUsados(escribir(1,hola),BS).
testBasico(8) :- buffersUsados(leer(12),[12]).
testBasico(9) :- buffersUsados(leer(10),BS), BS == [10].
testBasico(10) :- buffersUsados(computar,[]).
testBasico(11) :- buffersUsados(secuencia(escribir(10,azul),paralelo(escribir(4,rojo),leer(12))),[4,10,12]).
testBasico(12) :- buffersUsados(secuencia(escribir(15,blue),leer(15)),BS),BS==[15].
testBasico(13) :- buffersUsados(paralelo(paralelo(leer(t8),leer(a4)),escribir(a5,plp)),[a4,a5,t8]).
testBasico(14) :- buffersUsados(paralelo(leer(a1),leer(4)),BS) , BS == [4,a1].


cantidadTestsProcesos(11). % Actualizar con la cantidad de tests que entreguen
testProcesos(1) :- intercalar([],[],[]).
testProcesos(2) :- findall(XS,intercalar([1],[7,6],XS), YS), YS == [[1,7,6],[7,1,6],[7,6,1]].
testProcesos(3) :- intercalar([],[4,3,2,1],[4,3,2,1]).
testProcesos(4) :- intercalar([8,5,9,2],[],XS) , XS == [8,5,9,2].
testProcesos(5) :- intercalar([1,0,2],[5,9,3],[1,5,0,9,2,3]).
testProcesos(6) :- serializar(computar,[computar]).
testProcesos(7) :- serializar(leer(a9),XS), XS == [leer(a9)].
testProcesos(8) :- serializar(escribir(1,white),[escribir(1,white)]).
testProcesos(9) :- serializar(secuencia(escribir(10,black),leer(10)),XS),XS==[escribir(10,black),leer(10)].
testProcesos(10) :- serializar(paralelo(computar,leer(4)),[computar,leer(4)]).
testProcesos(11) :- findall(XS,serializar(paralelo(paralelo(leer(13),escribir(12,hola)),secuencia(computar,leer(5))),XS),YS), YS == [[leer(13), escribir(12, hola), computar, leer(5)] ,[leer(13), computar, escribir(12, hola), leer(5)] ,[leer(13), computar, leer(5), escribir(12, hola)] ,[computar, leer(13), escribir(12, hola), leer(5)] ,[computar, leer(13), leer(5), escribir(12, hola)] ,[computar, leer(5), leer(13), escribir(12, hola)] ,[escribir(12, hola), leer(13), computar, leer(5)] ,[escribir(12, hola), computar, leer(13), leer(5)],[escribir(12, hola), computar, leer(5), leer(13)] ,[computar, escribir(12, hola), leer(13), leer(5)] ,[computar, escribir(12, hola), leer(5), leer(13)] ,[computar, leer(5), escribir(12, hola), leer(13)]].


cantidadTestsBuffers(15). % Actualizar con la cantidad de tests que entreguen
testBuffers(1) :- contenidoBuffer(1,[escribir(1,hola),computar,computar,computar,escribir(1,chau)],[hola,chau]).
testBuffers(2) :- contenidoBuffer(1,[escribir(1,hola),computar,computar,computar,leer(1)],[]).
testBuffers(3) :- contenidoBuffer(1,[escribir(1,hola),computar,computar,computar,escribir(2,chau)],[hola]).
testBuffers(4) :- contenidoBuffer(1,secuencia(secuencia(escribir(1,hola),escribir(1,hola2)),escribir(1,hola3)),[hola,hola2,hola3]).
testBuffers(5) :- contenidoBuffer(1,paralelo(escribir(1,hola),escribir(2,chau)),[hola]).
testBuffers(6) :- findall(XS,contenidoBuffer(1,paralelo(escribir(1,hola),escribir(1,chau)),XS), YS), YS == [[hola,chau],[chau,hola]].
testBuffers(7) :- contenidoBuffer(1,paralelo(escribir(1,hola),computar),[hola]).
testBuffers(8) :- contenidoBuffer(1,[escribir(1,hola),computar,escribir(2,PLP),computar,escribir(3,TDA),escribir(5,LFAC),escribir(1,chau),computar,leer(1)],[chau]).
testBuffers(9) :- contenidoBuffer(1,secuencia(escribir(1,hola),leer(1)),[]).
testBuffers(10) :- contenidoLeido([escribir(1,hola),leer(1)],[hola]).
testBuffers(11) :- contenidoLeido([escribir(12,chau)],[]).
testBuffers(12) :- findall(C,contenidoLeido(paralelo(escribir(1,blue),secuencia(paralelo(leer(1),escribir(1,green)),leer(1))),C),XS), XS == [[blue,green],[blue,green],[green,blue], [green,blue]].
testBuffers(13) :- contenidoLeido(secuencia(escribir(4,juan),secuencia(leer(4),escribir(4,maria))),[juan]).
testBuffers(14) :- findall(C, contenidoLeido(paralelo(escribir(4,juan),paralelo(leer(4),escribir(4,maria))),C),XS), XS == [[juan],[juan],[maria],[maria]].
testBuffers(15) :- contenidoLeido([escribir(4,juan),leer(4),escribir(4,maria),leer(4)],[juan,maria]).


cantidadTestsSeguros(16). % Actualizar con la cantidad de tests que entreguen
testSeguros(1) :- esSeguro(computar).
testSeguros(2) :- esSeguro(secuencia(escribir(1,hola),secuencia(escribir(2,chau),secuencia(leer(1),leer(2))))).
testSeguros(3) :- esSeguro(paralelo(secuencia(secuencia(escribir(1,hola),escribir(2,chau)),secuencia(leer(1),leer(2))),secuencia(escribir(3,rosa),escribir(4,pink)))).
testSeguros(4) :- esSeguro(paralelo(secuencia(escribir(2,male),leer(2)),secuencia(escribir(1,oli),leer(1)))).
testSeguros(5) :- generarListaDeProcesos([computar],_,_).
testSeguros(6) :- generarListaDeProcesos([computar,escribir(1,a)],[1],[a]).
testSeguros(7) :- generarListaDeProcesos([computar,escribir(1,a),leer(1)],[1],[a]).
testSeguros(8) :- generarListaDeProcesos([computar,escribir(1,a),leer(1),escribir(2,b)],[1,2],[a,b]).
testSeguros(9) :- generarListaDeProcesos([computar,escribir(1,a),escribir(1,b),escribir(1,c),leer(1)],[1],[a,b,c]).
testSeguros(10) :- generarListaDeProcesos([computar,escribir(1,a),escribir(2,b),escribir(3,c),leer(1)],[1,2,3],[a,b,c]).
testSeguros(11) :- generarListaDeProcesos([computar,computar,computar,computar,computar,computar],[1,2],[a,b,c]).
testSeguros(12) :- generarListaDeProcesos([escribir(1,a),escribir(1,b),escribir(1,c),escribir(2,a),escribir(2,a),escribir(2,c)],[1,2],[a,b,c]).
testSeguros(13) :- generarListaDeProcesos([escribir(1,a),leer(1),escribir(1,b),leer(1),escribir(1,c),escribir(2,a),leer(2),escribir(2,a),leer(2),escribir(2,c)],[1,2],[a,b,c]).
testSeguros(14) :- esSeguro(paralelo(paralelo(escribir(5,azul),secuencia(escribir(4,lila),leer(4))),escribir(8,rosa))).
testSeguros(15) :- esSeguro(secuencia(paralelo(escribir(1,rojo),escribir(5,negro)),paralelo(escribir(5,verde),escribir(10,azul)))).
testSeguros(16) :- esSeguro(escribir(1,hola)).


tests(basico) :- cantidadTestsBasicos(M), forall(between(1,M,N), testBasico(N)).
tests(procesos) :- cantidadTestsProcesos(M), forall(between(1,M,N), testProcesos(N)).
tests(buffers) :- cantidadTestsBuffers(M), forall(between(1,M,N), testBuffers(N)).
tests(seguros) :- cantidadTestsSeguros(M), forall(between(1,M,N), testSeguros(N)).

tests(todos) :-
  tests(basico),
  tests(procesos),
  tests(buffers),
  tests(seguros).

tests :- tests(todos).