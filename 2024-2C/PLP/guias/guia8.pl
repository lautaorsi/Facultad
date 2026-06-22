juntar([], L2, L2).
juntar([X|L1], L2, [X|L3]) :- juntar(L1, L2, L3).

last([X],X).
last([_|XS],U) :- last(XS, U).

invertir([], []).
invertir([X|XS],C) :- invertir(XS, C1), juntar(C1, [X], C).

pertenece(X, [X|_]).
pertenece(X, [Y|XS]) :- not(Y = X) ,pertenece(X,XS).



agregarSiVale(X,L,C) :- member(X,L), C = L.
agregarSiVale(X,L,C) :- not(member(X,L)), append([X],L,C).



borrar([],_,[]).
borrar([X|XS],X,C) :- borrar(XS,X,C).
borrar([Y|XS],X,C) :- X \= Y , borrar(XS,X,C1), C = [Y|C1].

conjunto([],[]).
conjunto([X|XS],C) :- borrar(XS,X,C1), conjunto(C1, C2), C = [X|C2].






longitud([],0).
longitud([_|XS], C) :- longitud(XS,C1), C is C1 + 1.



desde(X,X).
desde(X,Y) :- N is X+1, desde(N,Y).


raiz(bin(_,R,_),R).

altura(nil,0).
altura(bin(A1,_,A2),C) :- altura(A1,C1), altura(A2,C2), C is max(C1,C2) + 1.

inorder(nil,[]).
inorder(bin(A1,X,A2),C) :- inorder(A1,C1),inorder(A2,C2), append(C1,[X],C3), append(C3,C2,C).


suman(X, Y, C) :-  between(0, C, X), Y is C - X.




listaQS(L,0,C):- length(C,L), todosCero(C) .
listaQS(L,N,C) :- between(1,N,X),length(C,L),L2 is L - 1, Y is N - X, listaQS(L2,Y, C1), sublista(C1,C), member(X,C).


sublista([X|XS],L2) :- cantAp(X,[X|XS],C), cantAp(X,L2,C).

filaSuma([X], X).
filaSuma([X|XS], N) :- between(0, N, X), Z is N-X, filaSuma(XS, Z).

cantAp(_,[],0).
cantAp(E,[E|XS],C) :- cantAp(E,XS, C1), C is C1 + 1.
cantAp(E,[X|XS],C) :- X \= E, cantAp(E,XS,C).  

%%numerosquesuman(?,?,?)
numerosquesuman(N1,N2,S) :- desde(0,S), between(0,S,N1), N2 is S - N1.

todosCero([0]).
todosCero([0|XS]) :- todosCero(XS).



pnp(N,Y) :- N2 is N + 1, desde(N2,Y),divisoresPrimos(Y,Divs),todosDividen(N,Divs).


divisoresPrimos(1,[1,1]).
divisoresPrimos(2,[1,2]).
divisoresPrimos(N,C) :- N2 is N - 1, between(2,N2,Y), 0 is N mod Y, divisoresPrimos(Y,[1,Y]), N3 is N / Y, divisoresPrimos(N3, C1), C = [Y|C1]. 

todosDividen(_,[]).
todosDividen(N,[X|XS]) :- 0 is N mod X, todosDividen(N,XS).




palabra(A,N,P) :- length(P,N), letrasEnAlfabeto(P,A).

letrasEnAlfabeto([],_).
letrasEnAlfabeto([X|XS],N) :- member(X,N), letrasEnAlfabeto(XS,N).


%% frase(A,F) :- desde(1,X), length(F,X), sonPalabras(F,A).

sonPalabras([],_).
sonPalabras([X|XS],A) :- length(X,Long), palabra(A,Long,X), sonPalabras(XS,A).

smlp([X|XS], P) :- length([X|XS], L1), between(0,L1,PL), length(P,PL),stm([X|XS],P),stp(P),contiguos(P,[X|XS]), not(hayMasLargo(P,[X|XS])).


hayMasLargo(P,L) :- smlp(L,P2), length(P2,L2), length(P,L1), L1 < L2.

stm(_,[]).
stm(P,[X|XS]) :- member(X,P), stm(P,XS).


stp([]).
stp([X|XS]) :- not(noesPrimo(X)), stp(XS).


noesPrimo(X) :- X2 is X - 1, between(2,X2,Y), 0 is X mod Y.

contiguos([_],_).
contiguos([X,Y|XS],P) :- nth0(PX,P,X), nth0(PY,P,Y), PY is PX + 1, contiguos([Y|XS], P).  

simbolos(S) :- member(S,[a,b]).

clausura(L) :- desde(1,X), length(L,X), tSimbolos(L).

tSimbolos([]).
tSimbolos([X|XS]) :- simbolos(X), tSimbolos(XS).



div(X,Y,0) :- X < Y.
div(X,Y,C) :- X >= Y, X2 is X - Y, div(X2,Y,C1), C is C1 + 1.

ochoReinas([X|XS]) :- CX is X mod 8, div(X,8,FX), DX is X mod 9, cd(CX,XS), fd(FX,XS), dd(DX,XS).

cd(_,[]).
cd(X,[Y|YS]) :- CY is Y mod 8, not(X = CY), cd(X,YS).

fd(_,[]).
fd(X,[Y|YS]) :- div(Y, 8, FY), not(X = FY), fd(X,YS).

dd(_,[]).
dd(X,[Y|YS]) :- DY is Y mod 9, not(X = DY), dd(X,YS).

elemmax([X],X).
elemmax([X|XS],C) :- elemmax(XS,C1), C is max(X,C1).

montana(L,L1,C,L2) :- elemmax(L,C), antesde(L,C,L1), reverse(L,RL), antesde(RL,C,RL2), reverse(RL2,L2).

antesde([C|_], C, [C]).
antesde([X|XS],C,L) :- not(X=C), antesde(XS,C,L1), L = [X|L1].






intercalar([],[],[]).
intercalar([X|XS],[],[X|XS]).
intercalar([],[X|XS],[X|XS]).
intercalar([X|XS],[Y|YS],[X,Y|L3]) :- intercalar(XS,YS,L3).
intercalar([X|XS],[Y|YS],[Y,X|L3]) :- intercalar(XS,YS,L3).

coprimos(X,Y) :- desde(1,X), between(1,X,Y), 1 is gcd(X,Y).


cuadradoSemiMagico(0,[]).
cuadradoSemiMagico(N,XS) :- not(N is 0), length(XS,N), listadeListasTamano(XS,N), desde(1,Num), listasQueSuman(XS,Num).


listadeListasTamano(Lista,N) :-length(Lista,Len), not((between(1,Len,I), nth1(I,Lista,X), not((length(X,N))))).

listasQueSuman(Lista,N) :- length(Lista,Len), not((between(1,Len,I), nth1(I,Lista,X), not((listaQueSuma(X,N))))).

listaQueSuma([],0).
listaQueSuma([X|XS],N) :- between(1,N,X), N2 is N-X, listaQueSuma(XS,N2).

listaQueSumaCualquier([]).
listaQueSumaCualquier(L) :- desde(1,N), listaQueSuma(L,N).

proxNumPoderoso(X,Y) :- X2 is X+1, desde(X2,Y), divsPrimos(Y,D), Dsqrd is D * D, 0 is Y mod Dsqrd, nohayMasChico(X,Y).

nohayMasChico(X,Y) :- X2 is X + 1, Y2 is Y - 1,not(( between(X2,Y2,Z), divsPrimos(Z,D), Dsqrd is D*D, 0 is Z mod Dsqrd, Z < Y)).

divsPrimos(X,Z) :- between(1,X,Z), esPrimo(Z),0 is X mod Z.


esPrimo(1).
esPrimo(X) :- X > 1, X2 is X - 1, not((between(2, X2, Z), X mod Z =:= 0)).


primos(X) :- desde(1,X), esPrimo(X).





paresQueSuman(S, X, Y) :- between(1, S, X), Y is S-X.
todosLosPares(X,Y) :- desde(1,S), between(1,S,X) , Y is S-X.








corteMasParejo(L,L1,L2) :- append(L1,L2,L), sum_list(L1,S1), sum_list(L2,S2), Coste is S1-S2,abs(Coste, AbsCoste), not(hayMasParejo(L,AbsCoste)).

hayMasParejo(L,OGCoste) :- append(L1,L2,L), sum_list(L1,S1), sum_list(L2,S2), Coste is S1-S2, abs(Coste, AbsCoste), AbsCoste < OGCoste.







generarCapicuas(L) :- desde(1,X), listaQueSuma(L,X), esCapicua(L).

esCapicua(L) :- reverse(L,RL), L = RL.



tokenizar(_,[],[]).
tokenizar(D,F,[X|T]) :- member(X,D), append(X,YS,F), tokenizar(D,YS,T).

subsecuenciaCreciente(_,[]).
subsecuenciaCreciente(L,S) :- esSubseq(L,S), esCreciente(S).

esSubseq([],[]).
esSubseq([X|XS],[X|S]) :- esSubseq(XS,S).
esSubseq([_|XS],S) :- esSubseq(XS,S).



esCreciente([_]).
esCreciente([X,Y|XS]) :- X =< Y, esCreciente([Y|XS]). 


subsecuenciaCrecienteMasLarga(L,S) :- subsecuenciaCreciente(L,S), not(hayMasLarga(L,S)).

hayMasLarga(L,S) :- subsecuenciaCreciente(L,S1), length(S1,LS1), length(S,LS), LS1 > LS.





frase(_,[]).
frase(XS, [F|FS]) :- esPalabra(XS,F), frase(XS,FS).


esPalabra(_,[]).
esPalabra(XS,[Y|YS]) :- member(Y,XS), esPalabra(XS,YS).





cantNodos(nil, 0).
cantNodos(bin(nil,_,nil),1).
cantNodos(bin(I,_,D), A) :- cantNodos(I,CI), cantNodos(D,CD), A is CD + CI + 1.



fruta(manzana).
fruta(limon).
fruta(pera).

dulce(manzana).
dulce(pera).

meGusta(helado).
meGusta(X) :- fruta(X), dulce(X).




caminosDesde(P,[P]).
caminosDesde(P,C) :- posiblePaso(P,P2), caminosDesde(P2,C1), append(P,C1,C).

posiblePaso((X,Y), (X1,Y)) :- X1 is X + 1.
posiblePaso((X,Y), (X1,Y)) :- X1 is X - 1.
posiblePaso((X,Y), (X,Y1)) :- Y1 is Y + 1.
posiblePaso((X,Y), (X,Y1)) :- Y1 is Y - 1.


esMasChico(_,[]).
esMasChico(D,[X|XS]) :- D =< X, esMasChico(D,XS).

todosDif([_],[]).
todosDif([X,Y|XS],DS) :- D is X - Y, todosDif([Y|XS],DS1), append([D], DS1, DS).


rotar(0,XS,XS).
rotar(N,[X|XS],RXS) :- N\=0,N1 is N - 1,append(XS,[X],RXS1),rotar(N1,RXS1,RXS).



juntar2([],YS,YS).
juntar2([X|XS],YS,[X|L]) :- juntar2(XS,YS,L).

prefijo(XS,X) :- append(X,_,XS).
sufijo(XS,X) :- append(_,X,XS).

sublista2(_,[]).
sublista2(XS,X) :- sufijo(XS,Y), prefijo(Y,X), X \= [].


aplanar([],[]).
aplanar([X|XS],[X|YS]) :- not(is_list(X)), aplanar(XS,YS).
aplanar([X|XS],YS) :- is_list(X), aplanar(X,Y), aplanar(XS,YS1), append(Y,YS1,YS).

partir(N,L,L1,L2) :- length(L,LL), prefijo(L,L1), length(L1,N), LR is LL - N, sufijo(L,L2), length(L2,LR).




caminosSimples(G,S,E,[S|P1]) :- member(A,G), tieneS(S,A,H) ,sacarElem(A,G,G1), caminosSimples(G1,H,E,P1).
caminosSimples(_,X,X,[X]).


tieneS(X,(X,Y),Y).
tieneS(X,(Y,X),Y).

sacarElem(_,[],[]).
sacarElem(X,[X|XS],P) :- sacarElem(X,XS,P).
sacarElem(X,[Y|XS], [Y|P]) :- X \= Y, sacarElem(X,XS,P).



aciclico(G) :- not((member((X,_),G),member((_,Y),G),hayCiclo(Y,G,[Y]) ,hayCiclo(X,G,[X]))).

hayCiclo(S,G,L) :- member(A,G),tieneS(S,A,N),member(N,L).
hayCiclo(S,G,L) :- member(A,G),tieneS(S,A,N), not(member(N,L)), append([N],L,L1), sacarElem(A,G,G1), hayCiclo(N,G1,L1).



corteparejo(L,L1,L2):- append(L1,L2,L), not(haymasparejo(L,L1,L2)).

haymasparejo(L,L1,L2) :- sumlist(L1,SL1), sumlist(L2,SL2), append(L3,L4,L), sumlist(L3,SL3),sumlist(L4,SL4), Dif1 is SL1 - SL2, Dif2 is SL3 - SL4, abs(Dif2) < abs(Dif1).


numpo(X,Y) :-X2 is X + 1, desde(X2,Y), not(hayprimoquenodivide(Y)), not(poderosoMasChico(X2,Y)), !.

poderosoMasChico(X,Y) :- between(X,Y,Z), not(hayprimoquenodivide(Z)),Z < Y.

hayprimoquenodivide(Y) :- primo(Y,P), P2 is P * P, Res is Y mod P2, 0 \= Res.

primo(_,1).
primo(Y,P) :- between(2,Y,P), esprimo(P), 0 is Y mod P.

esprimo(1).
esprimo(2).
esprimo(P) :- P2 is P - 1, not((between(2,P2,Z), 0 is P mod Z)).

arbol(A) :- desde(1,X), gda(A,X).

gda(nil,0).
gda(bin(X,_,Y), Z) :- Z > 0 , Z1 is Z - 1, between(0,Z1,Tx), between(0,Z1,Ty), Z1 is Tx + Ty, gda(X,Tx), gda(Y,Ty).


nodosEn(nil,_).
nodosEn(bin(I,X,D), XS) :- member(X,XS), nodosEn(I,XS), nodosEn(D,XS).


listasGemelas(L1,L2) :- desde(0,X), between(0,X,LL1), between(0,X,LL2), X is LL1 + LL2, listasDeLong(L1,LL1), listasDeLong(L2,LL2), append(L1,L2,L), reverse(L,L).

listasDeLong([],0).
listasDeLong([X|XS],L) :- L > 0 , L1 is L - 1, listasDeLong(XS,L1).

subcadenas([],[]).
subcadenas(Xs,[Y|Ys]) :- subcadenas(Xs,Ys).
subcadenas([X|Xs],[X|Ys]) :- subcadenas(Xs,Ys).
