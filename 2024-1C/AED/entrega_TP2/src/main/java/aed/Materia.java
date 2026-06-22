package aed;

import aed.SistemaSIU.CargoDocente;

public class Materia {
    // Invariante: el largo de listaAlumnos es igual a inscriptosMateria.
    // listaSinonimos no tiene tuplas duplicadas
    // El largo de plantelDocente es 4
    // listaAlumnos no tiene alumnos repetidos
    // Los int en plantelDocente y inscriptosMateria son mayores o iguales a 0

    private int[] plantelDocente;
    private ListaEnlazada<Tupla<Carrera, String>> listaSinonimos;
    private ListaEnlazada<String> listaAlumnos;
    private int inscriptosMateria;
    private DictTrie<String> conjuntoAlumnos;

    public Materia() {
        this.plantelDocente = new int[] { 0, 0, 0, 0 }; // O(1)
        this.listaSinonimos = new ListaEnlazada<Tupla<Carrera, String>>(); // O(1)
        this.listaAlumnos = new ListaEnlazada<String>(); // O(1)
        this.inscriptosMateria = 0; // O(1)
        this.conjuntoAlumnos = new DictTrie<String>();
    }

    public void agregarSinonimo(Carrera carrera, String nombreMateria) {
        Tupla<Carrera, String> nuevoSinonimo = new Tupla<Carrera, String>(carrera, nombreMateria); // O(1)
        this.listaSinonimos.agregarAtras(nuevoSinonimo); // O(1)
    }

    public void inscribirAlumno(String alumno) {
        this.inscriptosMateria += 1; // O(1)
        this.listaAlumnos.agregarAtras(alumno); // O(1)
        this.conjuntoAlumnos.agregar(alumno, alumno);
    }

    public void agregarDocente(CargoDocente cargo) { // O(1)
        switch (cargo) {
            case PROF:
                this.plantelDocente[0] += 1;
                break;
            case JTP:
                this.plantelDocente[1] += 1;
                break;
            case AY1:
                this.plantelDocente[2] += 1;
                break;
            case AY2:
                this.plantelDocente[3] += 1;
        }
    }

    public int[] plantelDocente() { // O(1)
        return plantelDocente;
    }

    public int inscriptos() { // O(1)
        return inscriptosMateria;
    }

    public boolean excedido() { // O(1)
        Boolean res;
        Boolean condProf = 250 * plantelDocente[0] < inscriptosMateria;
        Boolean condJTP = 100 * plantelDocente[1] < inscriptosMateria;
        Boolean condAY1 = 20 * plantelDocente[2] < inscriptosMateria;
        Boolean condAY2 = 20 * plantelDocente[3] < inscriptosMateria;
        res = (condProf || condJTP || condAY1 || condAY2);
        return res;
    }

    public DictTrie<String> devolverConjuntoAlumnos() {
        return this.conjuntoAlumnos;
    }

    public void cerrarMateria(DictTrie<Integer> alumnos) {
        Iterador<String> iterAlumnos = this.listaAlumnos.iterador(); // O(1)
        while (iterAlumnos.haySiguiente()) { // se ejecuta k veces
            String alumno = iterAlumnos.siguiente(); // O(1)
            int nMateriasAlumno = alumnos.devolver(alumno); // O(1)
            nMateriasAlumno -= 1; // O(1)
            alumnos.agregar(alumno, nMateriasAlumno); // O(1)
        } // O(1) + sum(k)(O(1)) = O(1) + k*O(1) = O(k). k = longitud de listaAlumnos =
          // cantidad de alumnos
          // = Em. Entonces O(k) = O(Em)

        Iterador<Tupla<Carrera, String>> iterSinonimos = this.listaSinonimos.iterador(); // O(1)
        while (iterSinonimos.haySiguiente()) { // se ejectuta h veces
            Tupla<Carrera, String> par = iterSinonimos.siguiente(); // O(1)
            Carrera carreraDestino = par.devolverPrimero(); // O(1)
            String nombreMateria = par.devolverSegundo(); // O(1)
            carreraDestino.remover(nombreMateria); // O(|m|)
        } // este while recorre listaSinonimos cuya longitud h indica la cantidad de
          // nombres de la materia
    } // y en cada iteracion se elimina c/u de esos nombres de sus respectivas
      // carreras (O(|m|)). Así que se puede
      // simplificar la complejidad como la sumatoria de la longitud de c/u de los
      // nombres de la materia
      // sum(n E Nm)(|n|)

}// cerrarMateria = O(Em + sum(n E Nm)(|n|))
