
--ejercicio 2
valorAbsoluto :: Float -> Float
valorAbsoluto x = sqrt(x*x)

bisiesto :: Int -> Bool
bisiesto x = mod x 4 == 0

factorial :: Int -> Int
factorial 0 = 1
factorial x = x * factorial (x-1)


--ejercicio 3
inverso :: Float -> Maybe Float
inverso 0 = Nothing
inverso x = Just(1/x)

entero :: Either Int Bool -> Int
entero (Left x) = x
entero (Right x) = if x then 1 else 0


--ejercicio 4

limpiar :: String -> String -> String
limpiar [] y = y
limpiar x [] = []
limpiar x (y:ys) | pertenece x y = limpiar x ys
                 | otherwise = y : limpiar x ys

pertenece :: String -> Char -> Bool
pertenece [] y = False
pertenece (x:xs) y | x == y = True
                   | otherwise = pertenece xs y
                  

difPromedio :: [Float] -> [Float]
difPromedio [] = []
difPromedio x = difpromedio2 (sum x / longitud x ) x



difpromedio2 :: Float -> [Float] -> [Float]
difpromedio2 x [] = []
difpromedio2 x (y:ys) = y - x : difpromedio2 x ys

longitud :: [a] -> Float
longitud [] = 0
longitud (x:xs) = 1.0 + longitud(xs)

todosIguales :: [Int] -> Bool
todosIguales [] = True
todosIguales (x:xs) | algunoDistinto x xs = False
                    | otherwise = todosIguales xs

algunoDistinto :: Int -> [Int] -> Bool
algunoDistinto x [] = False
algunoDistinto x (y:ys) = x /= y || algunoDistinto x ys