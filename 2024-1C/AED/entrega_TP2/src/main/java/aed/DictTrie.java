package aed;

import java.util.ArrayList;

public class DictTrie<V> { // Invariante: no se llega por 2 claves al mismo nodo
    // Si los nodos no tienen significado (val), tienen hijos.
    // No hay nodos huerfanos (no alcanzables por la raiz)
    // Los valores no son nulos si tienen una clave asociada
    // No hay claves vacías ni repetidas
    private Nodo raiz;
    private int cant_claves;

    private class Nodo {
        ArrayList<Nodo> sig; // O(1)
        String clave;
        V val; // O(1)
        int hijosNoNulo; // O(1)

        public Nodo() {
            sig = new ArrayList<Nodo>();
            // al ejecutarse siempre 255 veces tiene complejidad O(255) = O(1)
            for (int i = 0; i < 256; i++) {
                sig.add(null);
            }
            hijosNoNulo = 0; // O(1)
            val = null;
        }

    }

    public DictTrie() {
        raiz = new Nodo(); // O(1)
        cant_claves = 0;
    }

    public Boolean pertenece(String clave) {
        Nodo actual = raiz; // O(1)
        // este for tiene complejidad O(|s|) ya que en el peor caso cicla por todos los
        // caracteres del String
        for (int i = 0; i < clave.length(); i++) {
            int c = (int) clave.charAt(i); // O(1)
            if (actual.sig.get(c) == null) { // O(1)
                return false; // O(1)
            } else {
                actual = actual.sig.get(c); // O(1)
            }
        }
        return actual.val != null; // O(1)
    }

    public void agregar(String clave, V valor) {
        Nodo actual = raiz; // O(1)
        // este for tiene complegidad O(|s|) ya que siempre cicla por todos los
        // caracteres del String
        for (int i = 0; i < clave.length(); i++) {
            int c = (int) clave.charAt(i); // supongo que es O(1)
            if (actual.sig.get(c) == null) { // O(1)
                Nodo n = new Nodo();
                actual.sig.set(c, n); // O(1)
                actual.hijosNoNulo++; // O(1)
            }
            actual = actual.sig.get(c); // O(1)
        }
        actual.val = valor; // O(1)
        actual.clave = clave;
        cant_claves++;
    }

    public void quitar(String clave) {
        Nodo actual = raiz; // O(1)
        Nodo ultimoUtil = raiz; // O(1)
        int ultimoCUtil = clave.charAt(0); // O(1)
        // este for tiene complegidad O(|s|) ya que siempre cicla por todos los
        // caracteres del String
        for (int i = 0; i < clave.length(); i++) {
            int c = (int) clave.charAt(i); // O(1)
            if (actual.hijosNoNulo > 1) { // O(1)
                ultimoCUtil = c; // O(1)
                ultimoUtil = actual; // O(1)
            }
            actual = actual.sig.get(c); // O(1)
        }
        if (actual.hijosNoNulo == 0) { // O(1)
            ultimoUtil.sig.set(ultimoCUtil, null); // O(1)
        } else {
            actual.val = null; // O(1)
            actual.clave = null;
        }
        cant_claves --;
    }

    public V devolver(String clave) {
        Nodo actual = raiz; // O(1)
        for (int i = 0; i < clave.length(); i++) { // O(|s|) -> cicla por caracteres de string
            int c = (int) clave.charAt(i);
            if (actual.sig.get(c) == null) {
                return null;
            } else {
                actual = actual.sig.get(c);
            }
        }
        return actual.val;
    }

    public String[] listar() { // Se recorren todos los nodos del trie (O(n)) y se maneja array de tamaño k = cant claves
        //Complejidad = O(n+k)
        ArrayList<String> list = new ArrayList<String>(cant_claves); //k = cantidad de claves = O(k)

        list = listaRecursiva(list, raiz); //O(n)

        String[] res = new String[list.size()]; //O(k)
        res = list.toArray(res); //O(k) -> copia elementos de list a res
        return res;
    }

    private ArrayList<String> listaRecursiva(ArrayList<String> lista, Nodo actual) { //cada nodo se visita
        //una vez. O(n) = n, donde es cant de nodos no nulos
        if (actual.clave != null) { //O(1)
            lista.add(actual.clave); //O(1) ya que el ArrayList lo inicializamos con la cantidad de claves
            //evitando que se tenga que redimensionar para agregar elementos.
        }
        if (actual.hijosNoNulo > 0) { //O(1)
            for (int i = 0; i < 256; i++) { //O(255) = O(1)
                if (actual.sig.get(i) != null) { //O(1)
                    lista = listaRecursiva(lista, actual.sig.get(i)); //Se recorren todos los nodos no nulos
                }
            }
        }
        return lista;
    }
}