package aed;

public class Recordatorio {
    String global_message;
    Fecha fecha;
    Horario global_time;
    public Recordatorio(String mensaje, Fecha fecha, Horario horario) {
        this.global_message = mensaje;
        this.global_time = horario;
        this.fecha = new Fecha(fecha);
    }

    public Horario horario() {
        return this.global_time;
    }

    public Fecha fecha() {
        return new Fecha(fecha);
    }

    public String mensaje() {
        return this.global_message;
    }

    @Override
    public String toString() {
        return this.global_message + " @ " + this.fecha + " " + this.global_time;
    }

    @Override
    public boolean equals(Object otro) {
        if(otro.getClass() == this.getClass()){
            Recordatorio reminder = (Recordatorio) otro;
            return(this.global_message.equals(reminder.mensaje()) && this.fecha.equals(reminder.fecha()) && this.global_time.equals(reminder.horario()));
        }
        else{
            return(false);
        }
    }

}
