package aed;

public class SistemaSIU { // Invariante: no se puede repetir un mismo LU en una materia.
    // Cada alumno inscripto en una materia debe estar en trie alumnos
    // Cada alumno puede estar inscripto únicamente a materias de su/s carrera/s
    // Los nombres de las materias dentro de una carrera deben ser únicos
    // Todos los strings LU tienen una misma longitud maxima
    // cada alumno tiene un LU unico
    // Cada alumno aparece en tantas materias como indique su cantidad de materias
    // inscriptas, y estas deben
    // ser diferentes entre si
    // la cantidad de inscriptos en una materia debe ser menor o igual a la cantidad
    // de claves del trie alumnos
    private DictTrie<Carrera> carreras;
    private DictTrie<Integer> alumnos;

    enum CargoDocente {
        AY2,
        AY1,
        JTP,
        PROF
    }

    public SistemaSIU(InfoMateria[] infoMaterias, String[] libretasUniversitarias) {
        this.carreras = new DictTrie<Carrera>(); // O(1)
        this.alumnos = new DictTrie<Integer>(); // O(1)
        for (InfoMateria materia : infoMaterias) { // se ejecuta k veces (k=longitud infoMaterias = cantMaterias = |M|)
            Materia nuevaMateria = new Materia(); // O(1)
            for (ParCarreraMateria par : materia.getParesCarreraMateria()) { // ejecuta h veces (h=cant nombres de
                                                                             // materia =|Nm|por materia)
                String nombreCarrera = par.getCarrera(); // O(1)
                Carrera carreraDestino; // O(1)
                if (!(this.carreras.pertenece(nombreCarrera))) { // O(|c|)
                    carreraDestino = new Carrera(); // O(1)
                    carreras.agregar(nombreCarrera, carreraDestino); // O(|c|)
                } else {
                    carreraDestino = carreras.devolver(nombreCarrera); // O(|c|)
                }
                carreraDestino.agregar(par.getNombreMateria(), nuevaMateria); // O(|m|) ->para cada materia
                nuevaMateria.agregarSinonimo(carreraDestino, par.getNombreMateria()); // O(1)
            }
        }
        for (String alumno : libretasUniversitarias) { // ejecuta t veces (t=cantidad estudiantes=E)
            this.alumnos.agregar(alumno, 0); // O(1) -> strings LU acotados
        }
    } // O(E + sum(m E M)sum(n E Nm)(|n|) + sum(c E C)(|c|*|Mc|))

    public void inscribir(String estudiante, String carrera, String materia) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|) ->nombre de carrera
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|) -> nombre de materia
        if (!materiaDestino.devolverConjuntoAlumnos().pertenece(estudiante)) {
            materiaDestino.inscribirAlumno(estudiante); // O(1)
            int nMateriasAlumno = this.alumnos.devolver(estudiante); // O(1) -> porque LU son acotadas
            nMateriasAlumno += 1; // O(1)
            this.alumnos.agregar(estudiante, nMateriasAlumno); // O(1)
        }

    } // O(1) + O(|c|) + O(|m|) = O(|c| + |m|)

    // LISTO
    public void agregarDocente(CargoDocente cargo, String carrera, String materia) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|)
        materiaDestino.agregarDocente(cargo); // O(1)
    } // O(|c|+|m|)

    public int[] plantelDocente(String materia, String carrera) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|)
        return materiaDestino.plantelDocente(); // O(1)
    } // O(|c|+|m|)

    public void cerrarMateria(String materia, String carrera) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|)
        materiaDestino.cerrarMateria(this.alumnos); // O(Em + sum(n E Nm)(|n|))
    } // O(|c|+|m|+Em+sum(n E Nm)(|n|))

    public int inscriptos(String materia, String carrera) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|)
        return materiaDestino.inscriptos(); // O(1)
    } // O(|c|+|m|)

    public boolean excedeCupo(String materia, String carrera) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        Materia materiaDestino = carreraDestino.devolver(materia); // O(|m|)
        return materiaDestino.excedido(); // O(1)
    } // O(|c|+|m|)

    public String[] carreras() {
        return carreras.listar();
    }// O(sum(c E C)(|c|))

    public String[] materias(String carrera) {
        Carrera carreraDestino = this.carreras.devolver(carrera); // O(|c|)
        return carreraDestino.devolverMaterias().listar();
    }// O(|c| + sum(mc E Mc)(|mc|))

    public int materiasInscriptas(String estudiante) {
        return alumnos.devolver(estudiante); // O(1) -> debido a LU ACOTADA
    }
}
