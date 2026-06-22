data Arbol a = Hoja a | Bin (Arbol a) a (Arbol a)


--ej 2
foldAB :: (a -> b)-> (b -> a -> b -> b) -> Arbol a -> b
foldAB fHoja fBin (Hoja n) = fHoja n
foldAB fHoja fBin (Bin h1 n h2) = fBin (foldAB fHoja fBin h1) n (foldAB fHoja fBin h2)

esHoja :: Arbol a -> Bool
esHoja (Hoja n) = True
esHoja (Bin h1 n h2) = False

hojas :: Arbol a -> [a]
hojas = foldAB (\x -> [x]) (\a1 n a2 -> a1 ++ a2)

espejo :: Arbol a -> Arbol a
espejo = foldAB (\x -> Hoja x) (\a1 n a2 -> Bin a2 n a1)

ramas :: Arbol a -> [[a]]
ramas = foldAB (\x -> [[x]]) (\a1 n a2 ->   (map (n:) a1) ++ (map (n:) a2))

cantHojas :: Arbol a -> Integer
cantHojas = foldAB (\acc -> 1) (\x1 n x2 -> x1 + x2)






foldr2 :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
foldr2 f z [] [] = z
foldr2 f z (x:xs) (y:ys) = f x y (foldr2 f z xs ys)


data AT a = Nil | Tri a (AT a) (AT a) (AT a)


foldAT :: b -> (a -> b -> b -> b -> b) -> AT a -> b
foldAT fNil fTri Nil = fNil 
foldAT fNil fTri (Tri a x y z) = fTri a (foldAT fNil fTri x) (foldAT fNil fTri y) (foldAT fNil fTri z)

preorder :: AT a -> [a]
preorder = foldAT [] (\nodo t1 t2 t3 -> [nodo] ++ t1 ++ t2 ++ t3) 

mapAT :: (a -> b) -> AT a -> AT b
mapAT f = foldAT Nil (\nodo t1 t2 t3 -> Tri (f nodo) t1 t2 t3)

nivel :: AT a -> Int -> [a]
nivel = foldAT (const []) (\nodo t1 t2 t3 n ->   if n == 0 then [nodo] else t1 (n-1) ++ t2 (n-1) ++ t3 (n-1))

raiz :: AT a -> [a]
raiz Nil = []
raiz (Tri a t1 t2 t3) = [a]

