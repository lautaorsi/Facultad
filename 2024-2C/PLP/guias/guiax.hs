


suma :: [Int] -> Int
suma x = foldr (+) 0 x



conc ::  [t0] -> [t0] -> [t0]
conc x y = foldr (:) y x

filtr :: (t0 -> Bool) -> [t0] -> [t0]
filtr func [] = []
filtr func (y:ys) | func y = y: filtr func ys 
                  | otherwise = filtr func ys 

mp :: (t0 -> t1) -> [t0] -> [t1]
mp func [] = []
mp func (x:xs) = func x : mp func xs


--iii)
sumaParcial :: Num a => [a] -> [a]
sumaParcial [] = []
sumaParcial [x] = [x]
sumaParcial (x:y:xs) = x : sumaParcial(x+y:xs)

--iv) 
--basicamente, sumatoria de posiciones pares - sumatoria de posiciones impares
sumaAlternada :: [Int] -> Int
sumaAlternada [] = 0
sumaAlternada xs = foldr (+) 0 (posiciones 0 xs) - foldr (+) 0 (posiciones 1 xs)  

--posiciones recibe 2 params, un int y una seq, si el int es 0 agarra las posiciones pares, si es 1 las impares (si reciba otra cosa en el 1er param explota todo)
posiciones :: Int ->  [Int] -> [Int]
posiciones x [] = []
posiciones x [y] | x == 0 = [y]
                 | otherwise = []
posiciones x (y1:y2:ys) | x == 0 = y1 : posiciones 0  ys
                        | x == 1 = y2 : posiciones 1 ys
                        | otherwise = posiciones x (y1:y2:ys)



--Ej 10
--i
foldNat :: (Integer -> b -> b) -> b -> Integer -> b
foldNat _ z 0 = z
foldNat f z n =  f n (foldNat f z (n-1))

--ii
potencia :: Integer -> Integer -> Integer
potencia x y = foldNat (\_ acc -> x * acc) 1 y

--iii
factorial :: Integer -> Integer
factorial n = foldNat (\x acc -> (x) * acc) 1 n



data AB a = Nil | Bin (AB a) a (AB a) deriving Eq


foldAB :: b -> (b->a->b->b) -> AB a -> b 
foldAB fhoja fbin Nil  = fhoja
foldAB fhoja fbin (Bin i r d) = fbin (foldAB fhoja fbin i) r (foldAB fhoja fbin d)




data Operador = Sumar Int | DividirPor Int | Secuencia [Operador]

foldOperador :: (Int -> a) -> (Int -> a) -> ([a] -> a) -> Operador -> a
foldOperador cSumar _ _ (Sumar n) = cSumar n
foldOperador _ cDividir _ (DividirPor n) = cDividir n
foldOperador cSumar cDividir cSecuencia (Secuencia ns) = cSecuencia (map (foldOperador cSumar cDividir cSecuencia) ns)

falla :: Operador -> Bool
falla = foldOperador (const False) (==0) (elem True)


recr :: (a -> [a] ->b ->b ) ->b ->[a] ->b
recr f z [] = z
recr f z (x:xs) = f x xs (recr f z xs)


f :: [Int] -> Maybe (Int,Int)
f = (\xs -> if length xs <= 1 then Nothing else Just (foldr (\(x1,y1) (x2,y2) -> if x1 - y1 >= x2-y2 then (x2,y2) else (x1,y1)) (10000000,0) (diferencias xs)))

diferencias :: [Int] -> [(Int,Int)]
diferencias = recr (\x xs r -> if null xs then [] else [(x,head xs)] ++ r) [] 


foldr2 :: (a -> a -> b -> b) -> b -> [a] -> [a] -> b
foldr2 _ z _ [] = z
foldr2 _ z [] _ = z
foldr2 f z (x:xs) (y:ys) = f x y (rec xs ys) 
    where rec = foldr2 f z

f2 :: Eq a => a -> [a] -> [a] -> Bool
f2 e = foldr2 (\x y rx -> x == e || y == e || rx ) False


elem2 :: Eq a => a -> [a] -> [a] -> Bool
elem2 x xs ys = foldr (\(a, b) acc -> x == a || x == b || acc) False (zip xs ys)


mejorsegun :: (a -> a -> Bool) -> AB a -> a
mejorsegun f (Bin i r d) = foldAB r (\ri r rd -> if (f r ri) then (if f r rd then r else rd) else (if f ri rd then ri else rd)) (Bin i r d)   



mapPares :: (a -> b -> c) -> [(a,b)] -> [c]
mapPares f = map (\(x,y) -> f x y)


armarPares :: [a] -> [b] -> [(a,b)]
armarPares  = foldr  (\x r -> (\(y:ys) -> (x,y) : (r ys)))  (const [])

mapDoble :: (a -> b -> c) -> [a] -> [b] -> [c]
mapDoble f = foldr (\x r -> (\(y:ys) -> (f x y) : (r ys))) (const [])


data AT a = NilT | Tri a (AT a) (AT a) (AT a)

foldAT :: b -> (a -> b -> b -> b -> b) -> AT a -> b
foldAT z _ NilT = z
foldAT z f (Tri x ri rm rd) = f x (foldAT z f ri) (foldAT z f rm) (foldAT z f rd)

nivel :: AT a -> Int -> [a]
nivel = foldAT (const []) (\x ri rm rd -> (\n -> if n == 0 then [x] else ((ri (n-1)) ++ (rm (n-1)) ++ (rd (n-1)))))


asd :: Either a b -> Bool
asd (Left x) = True
asd (Right x) = False


minDif :: [Int] -> Maybe (Int,Int)
minDif = recr (\x xs (Just r) -> if xs == [] then (Just (0,100000000)) else (if (min (abs (x - (head xs))) (abs((fst r) - (snd r)))) == ((fst r) - (snd r)) then (Just r) else (Just (x, head xs)))) (Just (0,100000000))

recri :: b -> (a -> b -> [a] -> b) -> [a] -> b
recri z f xs = fst (foldr (\x (r,xs) -> (f x r xs, x:xs)) (z,[]) xs)