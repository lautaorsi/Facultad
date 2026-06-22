package aed;



// Todos los tipos de datos "Comparables" tienen el método compareTo()
// elem1.compareTo(elem2) devuelve un entero. Si es mayor a 0, entonces elem1 > elem2
public class ABB<T extends Comparable<T>> implements Conjunto<T> {
    private int cardinal;
    private Nodo raiz;
    private ABB<T> arb = this;


    private class Nodo {    
        private Nodo padre;
        private Nodo izquierdo;
        private Nodo derecho;
        private T valor;

        public Nodo(T v) {
            valor = v;
            izquierdo = null;
            derecho = null;
            padre = null;
        }

        public Nodo() {
            valor = null;
            izquierdo = null;
            derecho = null;
            padre = null;
        }



        private int Cantidaddescendientes(){
            int ret = 0;
            if(this.izquierdo != null){
                ret += 1;
            }
            if(this.derecho != null){
                ret += 1;
            }
            return ret;
        }


        //UltimoNodo retorna el nodo de valor mas cercano al "this" (por izq si tiene 2 desc, si tiene 1 entonces por el que tenga)
        private Nodo UltimoNodo(){
            Nodo nodo_actual;
            if(this.izquierdo != null){
                nodo_actual = this.izquierdo;
                while(nodo_actual.derecho != null){
                    nodo_actual = nodo_actual.derecho;
                }
            }
            else{
                nodo_actual = this.derecho;
                while(nodo_actual.izquierdo != null){
                    nodo_actual = nodo_actual.izquierdo;
                }
            }
            return nodo_actual;
        }

        //Printear retorna valores de forma recursiva, recorre unicamente los valores y los hijos NO LOS PADRES
        public String printear(){

            String res = "";
            if(this != null && this.izquierdo == null && this.derecho == null){
                res = this.valor + "";
            }
            if(this != null && this.izquierdo != null && this.derecho == null){
                res = this.izquierdo.printear() + "," + this.valor;
            }
            if(this != null && this.derecho != null && this.izquierdo == null){
                res = this.valor + "," + this.derecho.printear();
            }
            if(this != null && this.derecho != null && this.izquierdo != null){
                res = this.izquierdo.printear() + "," + this.valor + "," + this.derecho.printear();
            }
            return res;
        }

        //NodoCercano retorna el hijo de this si tiene, null si no tiene
        private Nodo NodoCercano(){
            Nodo ret;
            if(this.izquierdo != null){
                ret = this.izquierdo;
            }
            else{
                ret = this.derecho;
            }
            return ret;
        }






      

    }













    public ABB() {
        this.cardinal = 0;
        raiz = null;
    }



    public int cardinal() {
        return this.cardinal;
    }

    public T minimo(){
        Nodo nodoActual = raiz;
        while(nodoActual.izquierdo != null){
            nodoActual = nodoActual.izquierdo;
        }
        return nodoActual.valor;
    }

    public Nodo minimoNodo(){
        Nodo nodoActual = raiz;
        while(nodoActual.izquierdo != null){
            nodoActual = nodoActual.izquierdo;
        }
        return nodoActual;
    }

    public T maximo(){
        Nodo nodoActual = raiz;
        while(nodoActual.derecho != null){
            nodoActual = nodoActual.derecho;
        }
        return nodoActual.valor;
    };

    public void insertar(T elem){
        //si el arbol esta vacio
        if(cardinal == 0){
            this.raiz = new Nodo(elem);
            this.cardinal += 1;
        }
        //si el arbol no esta vacio
        else{
            this.cardinal += 1;
            //empezamos por la raiz
            Nodo nodoActual = raiz;

            //iteramos
            while(nodoActual != null){


                //si el valor del nodo actual es mayor:
                if((nodoActual.valor).compareTo(elem) > 0){
                    //chequeamos si tiene otro nodo mas pequeño
                    if(nodoActual.izquierdo == null){
                        //en caso de no tenerlo, le asignamos el valor que queremos insertar
                        Nodo nuevoNodo = new Nodo(elem);
                        nodoActual.izquierdo = nuevoNodo;
                        nuevoNodo.padre = nodoActual;
                        break;
                    }
                    else{
                        //en caso de tenerlo, recorremos el siguiente nodo mas pequeño
                        nodoActual = nodoActual.izquierdo;
                    }
                }



                //si el valor del nodo actual es menor
                if((nodoActual.valor).compareTo(elem) < 0){
                    //chequeamos si tiene otro nodo mas grande
                    if(nodoActual.derecho == null){
                        //en caso de no tenerlo, le asignamos el valor que queremos insertar
                        Nodo nuevoNodo = new Nodo(elem);
                        nodoActual.derecho = nuevoNodo;
                        nuevoNodo.padre = nodoActual;
                        break;
                    }
                    else{
                        //en caso de tenerlo, recorremos el siguiente nodo mas grande
                        nodoActual = nodoActual.derecho;
                        
                    }
                }
                //si ya pertenece, "cancelamos" la suma del principio de la funcion
                if((nodoActual.valor).compareTo(elem) == 0){
                    this.cardinal -= 1;
                    break;
                }
            }

        }


    }

    

    public boolean pertenece(T elem){
        boolean res = false;
        Nodo nodoActual = raiz;

        while(nodoActual != null){
            int MayorMenorIgual = (nodoActual.valor).compareTo(elem);
            if(MayorMenorIgual > 0){
                nodoActual = nodoActual.izquierdo;
            }
            if(MayorMenorIgual < 0){
                nodoActual = nodoActual.derecho;
            }
            if(MayorMenorIgual == 0){
                res = true;
                break;
            }

            
        }
        return res;






    }

    public void eliminar(T elem){
        //si no pertenece no hacemos nada
        if(this.pertenece(elem) && elem != null){
            this.cardinal -= 1;
            //empezamos por la raiz
            Nodo nodoActual = raiz;
            while(nodoActual != null){  

                //si el nodo actual es el que queremos eliminar
                if((nodoActual.valor).compareTo(elem) == 0){
                    //caso nodo != raiz
                    if(nodoActual.valor != raiz.valor){
                        if(nodoActual.Cantidaddescendientes() == 0){
                            Nodo NodoPadre = nodoActual.padre;
                            if(NodoPadre.izquierdo == nodoActual){
                                NodoPadre.izquierdo = null;
                            }
                            else{
                                NodoPadre.derecho = null;
                            }
                        }    
                        if(nodoActual.Cantidaddescendientes() == 1){
                            Nodo NodoPadre = nodoActual.padre;
                            if(NodoPadre.izquierdo == nodoActual){
                                NodoPadre.izquierdo = nodoActual.NodoCercano();
                            }
                            else{
                                NodoPadre.derecho = nodoActual.NodoCercano();
                            }
                            nodoActual.NodoCercano().padre = NodoPadre;
                        }
                        if(nodoActual.Cantidaddescendientes() == 2){
                            T NuevoValor = nodoActual.UltimoNodo().valor;
                            this.eliminar(nodoActual.UltimoNodo().valor);
                            this.cardinal += 1;
                            nodoActual.valor = NuevoValor;
                        }
                    }
                    //caso nodo == raiz
                    else{
                        if(nodoActual.Cantidaddescendientes() == 0){
                            raiz.valor = null;
                            raiz = null;
                        }
                        else if(nodoActual.Cantidaddescendientes() == 1){
                            raiz= nodoActual.NodoCercano();
                            nodoActual = null;
                        }
                        else if(nodoActual.Cantidaddescendientes() == 2){
                            T NuevoValor = nodoActual.UltimoNodo().valor;
                            this.eliminar(nodoActual.UltimoNodo().valor);
                            this.cardinal += 1;
                            nodoActual.valor = NuevoValor;
                        }
                    }




                    break;
                }


                
                //si no es el que queremos eliminar
                else{
                    //si es mayor vamos para la izquierda
                    if((nodoActual.valor).compareTo(elem) > 0){
                        nodoActual = nodoActual.izquierdo;
                    }
                    //caso contrario vamos a la derecha
                    else{
                        nodoActual = nodoActual.derecho;
                    }
                }
        }

    }
}




    public String toString(){
        return "{" + this.raiz.printear() + "}";
    }

    private class ABB_Iterador implements Iterador<T> {

        public boolean haySiguiente() {         
            return (arb.cardinal != 0);
        }
    

        public T siguiente() {
            T valorMinimo = arb.minimo();
            arb.eliminar(valorMinimo);
            return valorMinimo;
        }




    }

    public Iterador<T> iterador() {
        return new ABB_Iterador();
    }




}

//error: caso -99970 tiene 2 padres cuando hay que eliminarlo, el -130987 y -58971