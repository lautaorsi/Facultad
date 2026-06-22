package aed;

public class Horario {
    private int global_hour, global_minute, original_hour, original_minute;
    public Horario(int hora, int minutos) {
        global_minute = minutos;
        global_hour = hora;
        original_minute = minutos;
        original_hour = hora;
    }

    public int hora() {
        return this.global_hour;
    }

    public int minutos() {
        return this.global_minute;    
    }

    @Override
    public String toString() {
        return this.global_hour + ":" + this.global_minute;
    }

    @Override
    public boolean equals(Object otro) {
        if(otro.getClass() == this.getClass()){
            Horario otraFecha = (Horario) otro;
            if(otraFecha.minutos() == this.original_minute && otraFecha.hora() == this.original_hour){
                return(true);
            }
            else{
                return(false);
            }
        }
        else{
            return(false);
        }
    }





}
