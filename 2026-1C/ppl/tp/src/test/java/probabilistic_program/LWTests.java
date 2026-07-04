package probabilistic_program;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import utils.Tuple;

public class LWTests {

    @Test
    void literal() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram("5", new Random(42));
        assertEquals(5, t.first());
        assertEquals(0.0f, t.second());
    }

    @Test
    void letBinding() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram("(let (x 5) x)", new Random(42));
        assertEquals(5, t.first());
        assertEquals(0.0f, t.second());
    }

    @Test
    void ifExpression() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram("(if true 10 20)", new Random(42));
        assertEquals(10, t.first());
        assertEquals(0.0f, t.second());
    }

    @Test
    void sample() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram("(sample (normal 0 1))", new Random(42));
        assertEquals(0.0f, t.second()); // sampling shouldn't change log weight
    }

    @Test
    void observe() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram("(observe (normal 0 1) 0.5)", new Random(42));
        assertEquals(0.5f, (Float) t.first());
        assertEquals(-1.0439f, t.second(), 0.001f);
    }

    @Test
    void closureWithSample() {
        var t = ProbabilisticProgram.runLWProbabilisticProgram(
            "(let (f (fn (mu) (sample (normal mu 1)))) (f 5))", new Random(42));
        assertEquals(0.0f, t.second());
    }
}