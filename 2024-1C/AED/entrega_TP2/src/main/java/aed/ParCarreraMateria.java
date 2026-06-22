package aed;

public class ParCarreraMateria { //Invariante: los strings carrera y nombreMateria no son vacíos.
    String carrera;
    String nombreMateria;

    public ParCarreraMateria(String carrera, String nombreMateria) {
        this.carrera = carrera;
        this.nombreMateria = nombreMateria;
    }

    public String getNombreMateria() {
        return this.nombreMateria;
    }

    public String getCarrera() {
        return this.carrera;
    }
}
