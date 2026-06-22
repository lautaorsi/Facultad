module Proceso (Procesador, AT(Nil,Tern), RoseTree(Rose), Trie(TrieNodo), foldAT, foldRose, foldTrie, procVacio, procId, procCola, procHijosRose, procHijosAT, procRaizTrie, procSubTries, unoxuno, sufijos, 
inorder, preorder, postorder,
preorderRose, hojasRose, ramasRose, caminos, palabras, ifProc,(++!), (.!)) where



--Definiciones de tipos 

type Procesador a b = a -> [b]


 --Árboles ternarios 
data AT a = Nil | Tern a (AT a) (AT a) (AT a) deriving Eq
--E.g., at = Tern 1 (Tern 2 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 4 Nil Nil Nil)
--Es es árbol ternario con 1 en la raíz, y con sus tres hijos 2, 3 y 4.

 --RoseTrees
data RoseTree a = Rose a [RoseTree a] deriving Eq
--E.g., rt = Rose 1 [Rose 2 [], Rose 3 [], Rose 4 [], Rose 5 []] 
--es el RoseTree con 1 en la raíz y 4 hijos (2, 3, 4 y 5)

 --Tries
data Trie a = TrieNodo (Maybe a) [(Char, Trie a)] deriving Eq
--t = TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])]
 --es el Trie Bool de que tiene True en la raíz, tres hijos (a, b, y c), y, a su vez, b tiene como hijo a d.


-- Definiciones de Show

instance Show a => Show (RoseTree a) where
    show = showRoseTree 0
      where
        showRoseTree :: Show a => Int -> RoseTree a -> String
        showRoseTree indent (Rose value children) =
            replicate indent ' ' ++ show value ++ "\n" ++
            concatMap (showRoseTree (indent + 2)) children

instance Show a => Show (AT a) where
    show = showAT 0
      where
        showAT :: Show a => Int -> AT a -> String
        showAT _ Nil = replicate 2 ' ' ++ "Nil"
        showAT indent (Tern value left middle right) =
            replicate indent ' ' ++ show value ++ "\n" ++
            showSubtree (indent + 2) left ++
            showSubtree (indent + 2) middle ++
            showSubtree (indent + 2) right
        
        showSubtree :: Show a => Int -> AT a -> String
        showSubtree indent subtree =
            case subtree of
                Nil -> replicate indent ' ' ++ "Nil\n"
                _   -> showAT indent subtree

instance Show a => Show (Trie a) where
    show = showTrie ""
      where 
        showTrie :: Show a => String -> Trie a -> String
        showTrie indent (TrieNodo maybeValue children) =
            let valueLine = case maybeValue of
                                Nothing -> indent ++ "<vacío>\n"
                                Just v  -> indent ++ "Valor: " ++ show v ++ "\n"
                childrenLines = concatMap (\(c, t) -> showTrie (indent ++ "  " ++ [c] ++ ": ") t) children
            in valueLine ++ childrenLines


--Ejercicio 1
procVacio :: Procesador a b
procVacio _ = [] 

procId :: Procesador a a
procId x = [x]

procCola :: Procesador [a] a
procCola [] = []
procCola (x:xs) = xs

procHijosRose :: Procesador (RoseTree a) (RoseTree a)
procHijosRose (Rose a hijos) = hijos

procHijosAT :: Procesador (AT a) (AT a)
procHijosAT Nil = []
procHijosAT (Tern _ x y z) = [x,y,z] 

procRaizTrie :: Procesador (Trie a) (Maybe a)
procRaizTrie (TrieNodo raiz _) = [raiz]

procSubTries :: Procesador (Trie a) (Char, Trie a)
procSubTries (TrieNodo _ hijos) = hijos


--Ejercicio 2


foldAT::(a->b->b->b->b) -> b -> AT a -> b 
foldAT f base Nil = base
foldAT f base (Tern n h1 h2 h3) = f n (rec h1) (rec h2) (rec h3) 
    where rec = foldAT f base


foldRose :: (a -> [b] -> b) -> RoseTree a -> b
foldRose cRose (Rose n hijos) = cRose n (map rec hijos)
    where rec = foldRose cRose

foldTrie:: (Maybe a->[(Char, b)]-> b) -> Trie a -> b 
foldTrie f (TrieNodo raiz hijos) =  f raiz (map (\(char, b) -> (char, foldTrie f b)) hijos)


--Ejercicio 3
unoxuno :: Procesador [a] [a]
unoxuno = map (: [])

sufijos :: Procesador [a] [a] 
sufijos = foldr (\x accum -> (x:head accum): accum) [[]]


--Ejercicio 4

preorder :: Procesador (AT a) a 
preorder = foldAT  (\r recX recY recZ -> [r] ++ recX ++ recY ++ recZ) []

postorder :: Procesador (AT a) a
postorder = foldAT  (\r recX recY recZ -> recX ++ recY ++ recZ ++ [r]) []

inorder :: Procesador (AT a) a
inorder = foldAT  (\r recX recY recZ-> recX++ recY ++ [r] ++ recZ) []


--Ejercicio 5

preorderRose :: Procesador (RoseTree a) a
preorderRose = foldRose (\r recs -> r : concat recs)

hojasRose :: Procesador (RoseTree a) a
hojasRose = foldRose (\r recs-> if null recs then [r] else concat recs)

ramasRose :: Procesador (RoseTree a) [a]
ramasRose  = foldRose (\r recs -> if null recs then [[r]] else map(r:)(concat recs))


--Ejercicio 6

caminos :: Procesador (Trie a) String
caminos  trie = [] : foldTrie (\_ x ->  caminosAux x)  trie

caminosAux1:: Char -> [String] -> [String]
caminosAux1 a  lista = [a] :  map (a :) lista

caminosAux:: [(Char,[String])]-> [String]
caminosAux = concatMap (\x -> caminosAux1 (fst x) (snd x)) 


--Ejercicio 7
palabras :: Procesador (Trie a) String
palabras trie = let palabrasFormadas = foldTrie formarPalabras trie
              in filter (/= "") palabrasFormadas

formarPalabras :: Maybe a -> [(Char, [String])] -> [String]
formarPalabras val hijos = case val of
  Nothing -> foldr (\(c, subT) rec -> map(c:) subT ++ rec ) [] hijos
  Just a -> [] : foldr (\(c, subT) rec -> map(c:) subT ++rec ) [] hijos


--Ejercicio 8
 --8.a)
ifProc :: (a->Bool) -> Procesador a b -> Procesador a b -> Procesador a b
ifProc cond procTrue procFalse a = if cond a then procTrue  a else procFalse a

 --8.b)

(++!) :: Procesador a b -> Procesador a b -> Procesador a b
(++!) proc1 proc2 a = proc1 a ++ proc2 a

  --8.c)
(.!) :: Procesador b c -> Procesador a b -> Procesador a c
(.!) proc1 proc2 a = concat (map proc1 (proc2 a)) 




ejemploArbol :: RoseTree Int
ejemploArbol = Rose 1 [Rose 2 [Rose 7 [Rose 8 []]], Rose 3 [Rose 4 []], Rose 5 []]   
            
ejemploAt :: AT Int
ejemploAt = Tern 3 (Tern 2 Nil Nil Nil) (Tern 5 Nil Nil Nil) (Tern 7 (Tern 20 Nil Nil Nil) (Tern 25 Nil Nil Nil) (Tern 10 Nil Nil Nil))

atVacio :: AT Int
atVacio = Nil

ejemploTrie :: Trie Bool
ejemploTrie = TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])]

{-Tests-}

main :: IO Counts
main = do runTestTT allTests

allTests = test [  
  "ejercicio1" ~: testsEj1,
  "ejercicio2" ~: testsEj2,
  "ejercicio3" ~: testsEj3,
  "ejercicio4" ~: testsEj4,
  "ejercicio5" ~: testsEj5,
 "ejercicio6" ~: testsEj6,
  "ejercicio7" ~: testsEj7,
  "ejercicio8a" ~: testsEj8a,
  "ejercicio8b" ~: testsEj8b,
  "ejercicio8c" ~: testsEj8c
  ]

testsEj1 = test [  procVacio [] ~=? ([]::[Int])
   ,
   procVacio (Rose 1 [Rose 2 [], Rose 3 [], Rose 4 [], Rose 5 []])
     ~=? ([]::[Int])
   ,

   procId (Tern 1 (Tern 2 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 4 Nil Nil Nil))
     ~=? [Tern 1 (Tern 2 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 4 Nil Nil Nil)]
   ,
   procId []
     ~=? ([[]]::[[Int]])
   ,
   procId 1
     ~=? [1]
   ,

   procCola []
    ~=? ([]::[Int])
   ,
   procCola [1]
    ~=? ([]::[Int])
   ,
   procCola [Rose 1 []]
    ~=? ([]::[RoseTree Int])
   ,
   procCola [Rose 1[], Rose 2 []]
    ~=? [Rose 2 []]

   ,
   procHijosRose (Rose 1 [])
    ~=?([]::[RoseTree Int])
   ,
   procHijosRose (Rose 1 [Rose 2 [], Rose 3 []])
    ~=? [Rose 2 [], Rose 3 []]
   ,
   procHijosRose (Rose 1 [Rose 2 [Rose 5 [Rose 6 []]], Rose 3 [Rose 7 [Rose 8 [], Rose 9 []]]])
    ~=? [Rose 2 [Rose 5 [Rose 6 []]], Rose 3 [Rose 7 [Rose 8 [], Rose 9 []]]]

   ,
   procHijosAT (Tern 1 (Tern 2 Nil Nil Nil) Nil Nil)
    ~=? [Tern 2 Nil Nil Nil, Nil, Nil]
   ,
   procHijosAT (Tern 1 Nil Nil Nil)
    ~=? [Nil, Nil, Nil]
   ,
   procHijosAT (Tern 1 (Tern 2 (Tern 2 Nil Nil Nil) Nil Nil) (Tern 2 Nil (Tern 2 Nil Nil Nil) Nil) (Tern 2 Nil Nil (Tern 2 Nil Nil Nil)))
    ~=? [Tern 2 (Tern 2 Nil Nil Nil) Nil Nil, Tern 2 Nil (Tern 2 Nil Nil Nil) Nil, Tern 2 Nil Nil (Tern 2 Nil Nil Nil)]

   ,
   procRaizTrie (TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])])
    ~=? [Just True]
   ,
   procRaizTrie (TrieNodo (Just False) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])])
    ~=? [Just False]
   ,
   procRaizTrie (TrieNodo (Just 5) [])
    ~=? [Just 5]


   ,
   procSubTries (TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])])
    ~=? [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])]
   ,
   procSubTries (TrieNodo (Just True) [])
    ~=? []
   ,
   procSubTries (TrieNodo (Just True) [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])]),('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])])])
    ~=? [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])]),('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])])]
  
   ]


testsEj2 = [ 
  (foldAT (\r recI recC recD -> r+recI+recC+recD) 0 Nil) ~=? 0, --Suma de elems de Árbol Nil
  (foldAT(\r recI recC recD -> r+recI+recC+recD) 0 (Tern 4 (Nil) (Nil) (Nil))) ~=? 4,  --Suma de elems de árbol con 1 nodo(raiz)
  (foldAT(\r recI recC recD -> 5==r || recI || recC || recD) False (Tern 2 (Tern 1 (Nil) (Tern 5 (Nil) (Nil) (Nil)) (Nil)) (Tern 10 (Nil) (Nil) (Nil)) (Tern 4 (Nil) (Nil) (Nil))) ~=? True),--Verifica si el 5 pertenece al arbol 
  (foldRose(\r recs -> 1 + sum recs) (Rose 20 [])~=?1), --Cantidad de nodos de RoseTree de 1 elem (raiz)
  (foldRose (\r recs -> 1 + sum recs) ejemploArbol ~=? 7), --Cantidad de nodos de RoseTree
  (foldRose(\r recs -> r * product recs) (Rose 10 [Rose 5 [Rose 8 [Rose 4 []]], Rose 2 [Rose 1 []] ]) ~=? 3200), --Multiplicacion de elems de RoseTree
  (foldTrie (\v hijos -> 1 + sum(map snd hijos))) (TrieNodo Nothing []) ~=? 1, --Cantidad de nodos de Trie sin hijos
  (foldTrie (\v hijos -> 1 + sum(map snd hijos))) (TrieNodo Nothing [('A',TrieNodo (Just True) [('B',TrieNodo Nothing [])])]) ~=? 3, --Cant nodos de trie con hijos
  (foldTrie (\v hijos -> concatMap (\(char, t)-> [char]++t) hijos) (TrieNodo Nothing [('A',TrieNodo (Just True) [('R',TrieNodo (Just True) [('G',TrieNodo Nothing [])])])])) ~=? ['A','R','G'],
  (foldTrie (\v hijos -> concatMap (\(char, t)-> [char]++t) hijos) ejemploTrie )~=? ['a','b','a','d','c']] --Recopilación de todos los char del trie

testsEj3 = test [  
   unoxuno "hola"
     ~=? ["h","o","l","a"]
   ,
   unoxuno "hola "
     ~=? ["h","o","l","a", " "]
   ,
   unoxuno ""
     ~=? []
   ,
   unoxuno ["hola", "hola", "hola", "", " "]
     ~=? [["hola"],["hola"],["hola"],[""],[" "]]
   ,
   unoxuno [1]
     ~=? [[1]]
   ,
   sufijos "Plp"
    ~=? ["Plp", "lp", "p", ""]
   ,
   sufijos ""
    ~=? [""]
   ,
   sufijos "P"
    ~=? ["P", ""]
   ]

testsEj4 = test [
  (preorder Nil) ~=? ([]::[Int]),
  (preorder (Tern 5 (Tern 4 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 2 Nil Nil Nil)) ~=? [5,4,3,2]),
  (preorder ejemploAt) ~=? [3,2,5,7,20,25,10],
  (postorder Nil) ~=? ([]::[Int]),
  (postorder (Tern 5 (Tern 4 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 2 Nil Nil Nil)) ~=? [4,3,2,5]),
  (postorder ejemploAt) ~=? [2,5,20,25,10,7,3],
  (inorder Nil) ~=? ([]::[Int]),
  (inorder (Tern 5 (Tern 4 Nil Nil Nil) (Tern 3 Nil Nil Nil) (Tern 2 Nil Nil Nil)) ~=? [4,3,5,2]),
  (inorder ejemploAt) ~=? [2,5,3,20,25,7,10]                      
  ]

testsEj5 = test [ (preorderRose (Rose 1 []) )~=? [1],
  (preorderRose ejemploArbol) ~=? [1,2,7,8,3,4,5],
  (preorderRose (Rose 9 [Rose 3[],Rose 12 [],Rose 20 [Rose 10 []],Rose 8[]]))~=? [9,3,12,20,10,8],
  (hojasRose (Rose 3[]))~=? [3],
  (hojasRose ejemploArbol) ~=? [8,4,5],
  (hojasRose (Rose 9 [Rose 3[],Rose 12 [],Rose 20 [Rose 10 []],Rose 8[]]))~=? [3,12,10,8],
  (ramasRose (Rose 5[]))~=? [[5]],
  (ramasRose ejemploArbol) ~=? [[1,2,7,8],[1,3,4],[1,5]],
  (ramasRose (Rose 9 [Rose 3[],Rose 12 [],Rose 20 [Rose 10 []],Rose 8[]])) ~=? [[9,3],[9,12],[9,20,10],[9,8]]
  ]


testsEj6 = test [  
  caminos (TrieNodo Nothing [])        
    ~=? [""],                                           
  caminos (TrieNodo (Just True) [('a', TrieNodo (Just True) [])])
    ~=? ["", "a"],
  caminos (TrieNodo Nothing [('a', TrieNodo Nothing [])])
    ~=? ["", "a"],
  caminos (TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])])
    ~=? ["", "a", "b", "ba", "bad", "c"],
  caminos (TrieNodo Nothing [('h', TrieNodo Nothing [('o', TrieNodo Nothing [('l', TrieNodo Nothing [('a', TrieNodo (Just True) [])])])])])
    ~=? ["", "h", "ho", "hol", "hola"],
  caminos (TrieNodo (Just True) [('h', TrieNodo (Just True) [('o', TrieNodo (Just True) [('l', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])])])])
    ~=? ["", "h", "ho", "hol", "hola"]
  ]

testsEj7 = test [  
  palabras (TrieNodo Nothing [])         
    ~=? [],                                        
  palabras (TrieNodo (Just True) [])
    ~=? [],
  palabras (TrieNodo (Just True) [('a', TrieNodo (Just True) [])])
    ~=? ["a"],
  palabras (TrieNodo Nothing [('a', TrieNodo Nothing [])])
    ~=? [],
  palabras (TrieNodo (Just True) [('a', TrieNodo (Just True) []), ('b', TrieNodo Nothing [('a', TrieNodo (Just True) [('d', TrieNodo Nothing [])])]), ('c', TrieNodo (Just True) [])])
    ~=?["a", "ba", "c"],
  palabras (TrieNodo Nothing [('h', TrieNodo Nothing [('o', TrieNodo Nothing [('l', TrieNodo Nothing [('a', TrieNodo (Just True) [])])])])])
    ~=? ["hola"],
  palabras (TrieNodo (Just True) [('h', TrieNodo (Just True) [('o', TrieNodo (Just True) [('l', TrieNodo (Just True) [('a', TrieNodo (Just True) [])])])])])
    ~=? ["h", "ho", "hol", "hola"]
  ]

testsEj8a = test [ ifProc (\lista -> length lista>0) sufijos procVacio ([]::[String]) ~=? [] ,
  ifProc (\lista -> length lista>0) sufijos procVacio "Hola" ~=? ["Hola","ola","la","a",""],
  ifProc (\rose -> (foldRose (\r recs -> 1 + sum recs) rose > 1)) ramasRose procVacio (Rose 1[]) ~=? [], 
  ifProc (\rose -> (foldRose (\r recs -> 1 + sum recs) rose > 1)) ramasRose procVacio (Rose 1[Rose 5[],Rose 8[Rose 20[Rose 4[]]],Rose 2[Rose 11[]]]) ~=? [[1,5],[1,8,20,4],[1,2,11]],
  ifProc (\at -> at==Nil) procVacio preorder Nil ~=? ([]::[Int]),
  ifProc (\at -> at==Nil) procVacio preorder ejemploAt ~=? [3,2,5,7,20,25,10],
  ifProc (\t->((foldTrie (\v hijos -> 1 + sum(map snd hijos)) t > 1))) palabras caminos (TrieNodo Nothing []) ~=? [""],
  ifProc (\t->((foldTrie (\v hijos -> 1 + sum(map snd hijos)) t > 1))) palabras caminos ejemploTrie ~=? ["a","ba","c"]
  ]

testsEj8b = test [ 
  (++!) inorder preorder Nil  ~=? ([]::[Int]),
  (++!) inorder preorder ejemploAt ~=? [2,5,3,20,25,7,10,3,2,5,7,20,25,10],
  (++!) preorderRose hojasRose (Rose 7[]) ~=? [7,7],
  (++!) preorderRose hojasRose ejemploArbol ~=? [1,2,7,8,3,4,5,8,4,5],
  (++!) procVacio unoxuno [] ~=? ([]::[[Int]]),
  (++!) procVacio unoxuno [1,2,3,4,5,6,7,8] ~=? [[1],[2],[3],[4],[5],[6],[7],[8]],
  (++!) palabras caminos  ejemploTrie ~=? ["a","ba","c","","a","b","ba","bad","c"],
  (++!) palabras caminos  (TrieNodo Nothing []) ~=? [""]
  ]

testsEj8c = test [ 
  (.!) procVacio postorder Nil ~=? ([] :: [Int]), 
  (.!) procVacio postorder ejemploAt ~=? ([]::[Int]),
  (.!) procId ramasRose (Rose 4[]) ~=? [[4]],
  (.!) procId ramasRose ejemploArbol ~=? [[1,2,7,8],[1,3,4],[1,5]],
  (.!) unoxuno (\xs -> [map (+5) xs]) []~=? [],
  (.!) unoxuno (\xs -> [map(+5) xs]) [0,5,10,15,20,25,30,35,40] ~=? [[5],[10],[15],[20],[25],[30],[35],[40],[45]],
  (.!) sufijos palabras (TrieNodo Nothing []) ~=? [],
  (.!) sufijos palabras ejemploTrie ~=? ["a","","ba","a","","c",""]
  ]