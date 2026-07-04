# Funcionalidades


## Machine

### `initial_machine`
Retorna una máquina que parsea el código en una lista anidada de bloques de código, por ejemplo:
```clojure
    (let [suma (fn [x] (fn [y] (+ x y))) 
          f (suma 10)] 
     f 5)    
```

Es parseado a 

```python 
    [let, [suma, [fn, [x], [fn, [y], [+,x,y]]], f, [suma, 10]], [f, 5]]
```





