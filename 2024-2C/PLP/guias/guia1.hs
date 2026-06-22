import Data.List
curryT :: ((a, b) -> c) -> (a -> b -> c)
curryT  f x y = f (x,y)

unCurryT :: (a -> b -> c) -> ((a,b) -> c)
unCurryT f (x,y) = f x y

--3

sumfold :: Int -> [Int] -> Int
sumfold z xs = foldr (+) z xs 

elemfold :: Eq a => a -> [a] -> Bool 
elemfold z xs =  foldr (\x rec -> x == z || rec) False xs

concatfoldl :: [a] -> [a] -> [a]
concatfoldl ys xs = foldl (\rec x -> rec ++ [x] ) ys xs 

concatfoldr :: [a] -> [a] -> [a]
concatfoldr ys xs = foldr (\x rec -> rec ++ [x]) ys (reverse(xs))

filterfold :: [a] -> (a -> Bool) -> [a]
filterfold xs p = foldr (\x rec -> if p x then rec ++ [x] else rec) [] (reverse xs) 

mapfold :: [a] -> (a -> b) -> [b]
mapfold xs f = foldr(\x rec -> rec ++ [f x]) [] (reverse xs)

mejorSegun :: (a -> a -> Bool) -> [a] -> a
mejorSegun f xs = foldr1 (\x y -> if f x y then x else y) xs 

sumaAlt :: [Int] -> Int
sumaAlt xs = foldr1 (-) xs 

sumaAlt2 :: [Int] -> Int
sumaAlt2 = foldl1 (flip (-)) 

--5

entrelazar :: [a] -> [a] -> [a]
entrelazar (x:xs) = foldr (\x acc yys -> if null yys then x : acc [] else x : head yys : acc (tail yys)) (const []) (x:xs)
--vas desestimando la cola de yys con el const [] y solo agregas la cabeza de yys a medida que recorres con el foldr, 200 IQ

recr :: (a -> [a] -> b -> b) -> b -> [a] -> b
recr f z [] = z
recr f z (x : xs) = f x xs (recr f z xs)



--6


--8
mapPares :: (a ->b -> c) -> [(a,b)] -> [c]
mapPares f = foldr (\x rec -> (uncurry f x ) : rec ) []

armarPares:: [a] -> [b] -> [(a,b)]
armarPares = foldr (\x rec ys -> if  null ys then [] else ( x, head ys): rec (tail ys) )  (const []) 

mapDoble :: (a -> b -> c) -> [a] -> [b] -> [c]
mapDoble f = foldr (\x rec ys -> (f x (head ys)) : rec (tail ys)) (const [])  

--9

sumaMat :: [[Int]] -> [[Int]] -> [[Int]]
sumaMat = foldr (\x rec ys -> (mapDoble (+) x (head ys)) : rec (tail ys)) (const [])

data AB a = Nil | Bin (AB a) a (AB a)
--12
foldAB :: (a -> b -> b -> b) -> b -> AB a -> b
foldAB f z Nil = z
foldAB f z (Bin rd x ri) = f x (foldAB f z rd) (foldAB f z ri)  

recAB :: (a -> AB a -> AB a -> b -> b -> b) -> b -> AB a -> b
recAB f z Nil = z
recAB f z (Bin rd x ri) = f x rd ri (recAB f z rd) (recAB f z ri) 

esNil :: AB a -> Bool
esNil Nil = True
esNil _ = False

altura :: AB a -> Int
altura = foldAB (\_ ri rd -> (max ri rd) + 1) 0 

cantNodos :: AB a -> Int
cantNodos = foldAB (\_ ri rd -> ri + rd + 1) 0

mejorSegunAB :: (a -> a -> Bool) -> AB a -> a
mejorSegunAB f (Bin izq r der) = foldAB (\x ri rd -> if f r ri && f r rd then r else if f ri rd then ri else rd) r (Bin izq r der)

--13

cantHojas :: AB a -> Int
cantHojas = foldAB (\x ri rd -> if ri == 0 && rd == 0 then 1 else ri + rd) 0  

espejo :: AB a -> AB a
espejo = foldAB (\x ri rd -> Bin rd x ri) Nil



--15
data RoseTree a = Rose a [RoseTree a]

foldRose ::  (a -> [b] -> b) -> RoseTree a -> b
foldRose f (Rose x ys) = f x (map (foldRose f) ys) 

distancias :: RoseTree a -> [(a,Int)] 
distancias =  foldRose (\r recHijos -> if null recHijos then [(r,0)] else map (\(tree, distance) -> (tree, distance + 1)) (concat recHijos))

alturaRose :: RoseTree a -> Int
alturaRose = foldRose (\r recHijos -> if null recHijos then 1 else maximum recHijos + 1)


data Buffer a = Empty | Write Int a (Buffer a) | Read Int (Buffer a)

foldBuffer :: b -> (Int -> a -> b -> b) -> (Int -> b -> b) -> Buffer a -> b
foldBuffer cEmpty cWrite cRead Empty = cEmpty
foldBuffer cEmpty cWrite cRead (Write n x b) = cWrite n x (foldBuffer cEmpty cWrite cRead b)
foldBuffer cEmpty cWrite cRead (Read n b) = cRead n (foldBuffer cEmpty cWrite cRead b)

contenidoBuffer :: Int -> Buffer a -> Maybe a
contenidoBuffer e = foldBuffer Nothing (\n x rec -> if n == e then Just x else rec) (\_ rec -> rec)

posicionesOcupadas :: Buffer a -> [Int]
posicionesOcupadas = foldBuffer [] (\x _ rec -> union [x] rec) (\y rec -> delete y rec)

recBuff :: b -> (Int -> a -> Buffer a -> b -> b) -> (Int -> Buffer a -> b -> b) -> Buffer a -> b
recBuff cEmpty _ _ Empty = cEmpty
recBuff cEmpty cWrite cRead (Write a c buff) = cWrite a c buff (recBuff cEmpty cWrite cRead buff)
recBuff cEmpty cWrite cRead (Read a buff) = cRead a buff (recBuff cEmpty cWrite cRead buff)




data AT a = NilT | Tri a (AT a) (AT a) (AT a)


foldAT :: b -> (a -> b -> b -> b -> b) -> AT a -> b
foldAT z _ NilT = z
foldAT z f (Tri r i m d) = f r (foldAT z f i) (foldAT z f m) (foldAT z f d) 

preorderAT :: AT a -> [a]
preorderAT = foldAT [] (\r i m d -> [r] ++ i ++ m ++ d)
  
mapAT :: (a -> b) -> AT a -> AT b
mapAT f = foldAT NilT (\r i m d -> Tri (f r) i m d)

nivelAT :: AT a -> Int -> [a]
nivelAT  = foldAT (const []) (\r ri rm rd -> \i -> if i == 0 then [r] else ((ri (i-1)) ++ (rm (i-1)) ++ (rd (i-1))))



data AIH a = Hoja a | BinA (AIH a) (AIH a)

foldAIH :: (a -> b) -> (b -> b -> b) -> AIH a -> b
foldAIH fHoja _ (Hoja a) = fHoja a
foldAIH fHoja fBin (BinA i d) = fBin (foldAIH fHoja fBin i) (foldAIH fHoja fBin d)

alturaAIH :: AIH a -> Int
alturaAIH = foldAIH (const 1) (\a1 a2 -> 1 + max a1 a2)

tamano :: AIH a -> Int
tamano = foldAIH (const 1) (\a1 a2 -> 1 + a1 + a2)



data DatoX a = DatoX a a a


trim :: [Char] -> [Char]
trim = foldr (\x rec -> if x == ' ' then rec else x:rec) [] 

sacarUna :: Eq a => a -> [a] -> [a]
sacarUna = (\x ->  recr (\y ys rec -> if x == y then ys else y:rec) []  ) 

insertarOrdenado :: Ord a => a -> [a] -> [a]
insertarOrdenado = (\x -> recr (\y ys rec -> if x < y then x:y:ys else y : (if null rec then [x] else rec)) [])

completarLectura :: Buffer a -> Bool
completarLectura = recBuff True (\y _ ys rec -> rec) (\y ys rec -> elem y (posicionesOcupadas ys) && rec)   

 
data Prop = Var String | No Prop | Y Prop Prop | O Prop Prop | Imp Prop Prop 

type Valuacion = String -> Bool


foldProp :: (String -> b) -> (b -> b) -> (b -> b -> b) -> (b -> b -> b) -> (b -> b -> b) -> Prop -> b
foldProp cVar _ _ _ _ (Var x) = cVar x
foldProp cVar cNo cY cO cImp (No x) = cNo (foldProp cVar cNo cY cO cImp x)
foldProp cVar cNot cY cO cImp (Y x z) = cY (foldProp cVar cNot cY cO cImp x)  (foldProp cVar cNot cY cO cImp z )
foldProp cVar cNot cY cO cImp (O x z) = cO (foldProp cVar cNot cY cO cImp x)  (foldProp cVar cNot cY cO cImp z )
foldProp cVar cNot cY cO cImp (Imp x z) = cImp (foldProp cVar cNot cY cO cImp x)  (foldProp cVar cNot cY cO cImp z )

recProp :: (String -> b) -> (Prop -> b -> b) -> (Prop ->Prop ->b -> b -> b) -> (Prop ->Prop ->b -> b -> b) -> (Prop ->Prop ->b -> b -> b) -> Prop -> b
recProp cVar _ _ _ _ (Var x) = cVar x
recProp cVar cNo cY cO cImp (No x) = cNo x (recProp cVar cNo cY cO cImp x)
recProp cVar cNot cY cO cImp (Y x z) = cY x z (recProp cVar cNot cY cO cImp x)  (recProp cVar cNot cY cO cImp z )
recProp cVar cNot cY cO cImp (O x z) = cO x z (recProp cVar cNot cY cO cImp x)  (recProp cVar cNot cY cO cImp z )
recProp cVar cNot cY cO cImp (Imp x z) = cImp x z (recProp cVar cNot cY cO cImp x)  (recProp cVar cNot cY cO cImp z )



evaluar :: Valuacion -> Prop -> Bool
evaluar = (\f -> foldProp (\x -> f x) (\x -> not x) (\x y -> x && y) (\x y -> x || y) (\x y -> (not x) || y))

esVar :: Prop -> Bool
esVar = foldProp (const True) (const False) (\x y -> False) (\x y -> False) (\x y -> False)


estaenFnn :: Prop -> Bool
estaenFnn = recProp (const True) (\x rx -> esVar(x) && rx) (\_ _ rx ry -> rx && ry) (\_ _ rx ry -> rx && ry) (\_ _ _ _ -> False) 

data Melodia = Silencio Int | Nota Int Int | Paralelo [Melodia]

foldMelodia :: (Int -> a) -> (Int ->Int -> a) -> ([a] -> a) -> Melodia -> a
foldMelodia cSilencio _ _ (Silencio n) = cSilencio n
foldMelodia cSilencio cNota _ (Nota b n) = cNota b n
foldMelodia cSilencio cNota cParalelo (Paralelo xs) = cParalelo (map (foldMelodia cSilencio cNota cParalelo) xs )

duracionTotal :: Melodia -> Int
duracionTotal = foldMelodia id (\_ n -> n) (maximum)

data Operador = Sumar Int | DividirPor Int | Secuencia [Operador]

foldOperator :: (Int -> a) -> (Int -> a) -> ([a] -> a) -> Operador -> a
foldOperator cSumar _ _ (Sumar n) = cSumar n
foldOperator _ cDiv _ (DividirPor n) = cDiv n
foldOperator cSuma cDiv cSec (Secuencia ns) = cSec (map (foldOperator cSuma cDiv cSec) ns) 

aplanar :: Operador -> Operador
aplanar = foldOperator (Sumar) (DividirPor) (\xs -> Secuencia (foldr (:) [] xs) )