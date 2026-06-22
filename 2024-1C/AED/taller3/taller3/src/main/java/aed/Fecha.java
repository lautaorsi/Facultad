package aed;



public class Fecha {
    private int dia, mes;
    public Fecha(int dia, int mes) {
        this.dia = dia;
        this.mes = mes;
    }

    public Fecha(Fecha fecha) {
        this.dia = fecha.dia();
        this.mes = fecha.mes();
    }

    public Integer dia() {
        return dia;
    }

    public Integer mes() {
        return mes;
    }

    public String toString() {
        String sbuffer =  dia + "/" + mes;
        return sbuffer;
    }

    @Override
    public boolean equals(Object otra) {
        if(otra.getClass() == this.getClass()){
            if(((Fecha) otra).mes() == this.mes() && ((Fecha) otra).dia() == this.dia()){
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

    public void incrementarDia() {
        if(dia <= diasEnMes(mes)){
            dia += 1;
        }
        if(dia > diasEnMes(mes) && mes < 11){
            mes += 1;
            dia = 1;
        };
        if(dia > diasEnMes(mes) && mes == 12){
            mes = 1;
            dia = 1;
        }
    }

    private int diasEnMes(int mes) {
        int dias[] = {
                // ene, feb, mar, abr, may, jun
                31, 28, 31, 30, 31, 30,
                // jul, ago, sep, oct, nov, dic
                31, 31, 30, 31, 30, 31
        };
        return dias[mes - 1];
    }





}