package aed;

public class Carrera {
    private DictTrie<Materia> materias;

    public Carrera() {
        this.materias = new DictTrie<Materia>(); // O(1)
    }

    public void agregar(String nombreMateria, Materia materia) {
        materias.agregar(nombreMateria, materia); // O(|m|)
    }

    public Materia devolver(String nombreMateria) {
        return materias.devolver(nombreMateria); // O(|m|)
    }

    public void remover(String nombreMateria) {
        this.materias.quitar(nombreMateria); // O(|m|)
    }

    public DictTrie<Materia> devolverMaterias() {
        return this.materias;
    }

}