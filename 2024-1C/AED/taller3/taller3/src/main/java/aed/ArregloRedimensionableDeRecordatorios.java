package aed;



import java.util.Arrays;

class ArregloRedimensionableDeRecordatorios implements SecuenciaDeRecordatorios {

    private Recordatorio[] arr;
    private int contador = 0;

    public ArregloRedimensionableDeRecordatorios() {
        this.arr = new Recordatorio[0];
        this.contador = 0;
    }

    public ArregloRedimensionableDeRecordatorios(ArregloRedimensionableDeRecordatorios vector) {
        this.arr = new Recordatorio[vector.longitud() + 1];
        System.arraycopy(vector.arr, 0, this.arr, 0, vector.arr.length);
        this.contador = vector.contador;
    }

    public int longitud() {
        return this.contador;
    }

    public void agregarAtras(Recordatorio i) {
        Recordatorio[] newArr = new Recordatorio[this.arr.length + 1];
        System.arraycopy(this.arr, 0, newArr, 0, this.arr.length);
        newArr[newArr.length - 1] = i;
        this.arr = newArr;
        this.contador += 1;
    }

    public Recordatorio obtener(int i) {
        return this.arr[i];
    }

    public void quitarAtras() {
        Recordatorio[] newArr = new Recordatorio[arr.length - 1];
        for(int i = 0; i< newArr.length; i++){
            newArr[i] = arr[i];
        }
        this.arr = newArr;
        this.contador -= 1;
    }

    public void modificarPosicion(int indice, Recordatorio valor) {
        this.arr[indice] = valor;
    }

    public ArregloRedimensionableDeRecordatorios copiar() {
        ArregloRedimensionableDeRecordatorios copia = new ArregloRedimensionableDeRecordatorios();
        Recordatorio[] dummyarr =  new Recordatorio[this.arr.length];
        System.arraycopy(this.arr, 0, dummyarr, 0, this.arr.length);;
        copia.arr = dummyarr;
        copia.contador =  this.contador;
        return copia;
    }


}
