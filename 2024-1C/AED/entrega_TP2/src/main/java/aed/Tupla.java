package aed;

public class Tupla<A, B> {
    private A primero;
    private B segundo;

    public Tupla(A primero, B segundo) {
        this.primero = primero;
        this.segundo = segundo;
    }

    public A devolverPrimero() {
        return primero;
    }

    public B devolverSegundo() {
        return segundo;
    }
}