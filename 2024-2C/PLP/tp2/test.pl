long([],0).
long([X|XS], L) :- long(XS,L2), L is L2+1.