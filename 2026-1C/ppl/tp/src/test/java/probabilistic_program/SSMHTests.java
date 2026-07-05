package probabilistic_program;

import java.util.Random;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SSMHTests {

    @Test
    void singleSampleNoObserve(){
        // Exactly one sample site — the minimal valid program for SSMH.
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(
            "(sample (normal 0 1))", new Random(1), 5000, 500);
        assertEquals(5000, chain.size());
        for(Float v : chain) assertTrue(Float.isFinite(v));
    }

    @Test
    void sampleOnlyPriorOnly(){
        // Sample from N(0,1), no observe — chain should hover around the prior mean (0),
        // since there's nothing pulling it elsewhere. Loose bound since it's still MCMC noise.
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(
            "(let (mu (sample (normal 0 1))) mu)", new Random(2), 20000, 2000);
        float mean = 0f;
        for(Float v : chain) mean += v;
        mean /= chain.size();
        assertEquals(0.0f, mean, 0.1f);
    }

    @Test
    void conjugateSingleObserve(){
        // Notebook's "conj" example — exact posterior mean 1.150, std sqrt(0.5) ≈ 0.707
        String conj = "(let (mu (sample (normal 0 1))) (observe (normal mu 1) 2.3) mu)";
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(conj, new Random(0), 60000, 3000);

        float mean = 0f;
        for(Float v : chain) mean += v;
        mean /= chain.size();

        assertEquals(1.150f, mean, 0.05f);
    }

    @Test
    void conjugateTwoObserves(){
        // Same prior, two observations (1.0 and 3.0) instead of one.
        // Posterior precision = prior(1) + 2*likelihood(1) = 3; mean = (0*1 + (1+3)*1)/3 = 1.333
        String twoObs = "(let (mu (sample (normal 0 1))) (observe (normal mu 1) 1.0) (observe (normal mu 1) 3.0) mu)";
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(twoObs, new Random(5), 60000, 3000);

        float mean = 0f;
        for(Float v : chain) mean += v;
        mean /= chain.size();

        assertEquals(1.333f, mean, 0.05f);
    }

    @Test
    void chainLengthExcludesWarmup(){
        // Sanity check on bookkeeping: chain.size() should equal steps, not steps+warmup.
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(
            "(let (mu (sample (normal 0 1))) (observe (normal mu 1) 2.3) mu)",
            new Random(3), 500, 100);
        assertEquals(500, chain.size());
    }
}