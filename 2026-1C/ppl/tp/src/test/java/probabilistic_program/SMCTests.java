package probabilistic_program;

import java.util.ArrayList;
import java.util.Random;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SMCTests {

    private ArrayList<Random> makeRngs(int n, long seedBase){
        ArrayList<Random> rngs = new ArrayList<>();
        for(int i = 0; i < n; i++) rngs.add(new Random(seedBase + i));
        return rngs;
    }

    @Test
    void literalNoEffects(){
        // No sample/observe at all — every particle should just return the same literal.
        var results = ProbabilisticProgram.runSMCProbabilisticProgram("5", makeRngs(4, 1), 4);
        for(Float v : results) assertEquals(5.0f, v);
    }

    @Test
    void sampleOnlyNoObserve(){
        // Sample but no observe — advance() auto-answers the sample, so particles
        // go straight to 'done' with no resampling round ever triggered.
        var results = ProbabilisticProgram.runSMCProbabilisticProgram(
            "(sample (normal 0 1))", makeRngs(5, 10), 5);
        assertEquals(5, results.size());
        // values should differ across particles (independent rngs), but all finite
        for(Float v : results) assertTrue(Float.isFinite(v));
    }

    @Test
    void singleObserveTriggersResampling(){
        // One observe -> exactly one resampling round before 'done'.
        var results = ProbabilisticProgram.runSMCProbabilisticProgram(
            "(let (x (sample (normal 0 1))) (observe (normal x 1) 2.0) x)",
            makeRngs(50, 100), 50);
        assertEquals(50, results.size());
        // posterior mean should be pulled toward 2.0 relative to the N(0,1) prior
        float mean = 0f;
        for(Float v : results) mean += v;
        mean /= results.size();
        assertTrue(mean > 0.3f); // loose sanity bound, not exact
    }

    @Test
    void multipleSequentialObserves(){
        // Two observes -> two resampling rounds; exercises repeated fork()/rng reuse.
        var results = ProbabilisticProgram.runSMCProbabilisticProgram(
            "(let (x (sample (normal 0 1))) (observe (normal x 1) 1.5) (observe (normal x 1) 2.5) x)",
            makeRngs(50, 200), 50);
        assertEquals(50, results.size());
        for(Float v : results) assertTrue(Float.isFinite(v));
    }

    @Test
    void closureWithSampleAndObserve(){
        // Combines everything: closure application, sample inside a call, one observe.
        var results = ProbabilisticProgram.runSMCProbabilisticProgram(
            "(let (f (fn (mu) (sample (normal mu 1)))) (let (x (f 0)) (observe (normal x 1) 3.0) x))",
            makeRngs(30, 300), 30);
        assertEquals(30, results.size());
        for(Float v : results) assertTrue(Float.isFinite(v));
    }
}