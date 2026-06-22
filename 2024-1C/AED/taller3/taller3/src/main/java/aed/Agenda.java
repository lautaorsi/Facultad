package aed;

public class Agenda {

    Fecha fecha;
    ArregloRedimensionableDeRecordatorios arreglo = new ArregloRedimensionableDeRecordatorios();

    public Agenda(Fecha fechaActual) {
        this.fecha = new Fecha(fechaActual);
    }

    public void agregarRecordatorio(Recordatorio recordatorio) {
        this.arreglo.agregarAtras(recordatorio);
    }

    @Override
    public String toString() {
        String text = "";
        text += this.fecha + "\n=====\n" ;
        for(int i = 0; i < arreglo.longitud(); i++){
            if((arreglo.obtener(i).fecha()).equals(this.fecha) ){
                text += arreglo.obtener(i) + "\n";
            }
        }
        return text;
        

    }

    public void incrementarDia() {
        this.fecha.incrementarDia();;
    }

    public Fecha fechaActual() {
        return this.fecha;
    }




}
