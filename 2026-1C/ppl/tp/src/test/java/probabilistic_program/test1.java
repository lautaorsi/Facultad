package probabilistic_program;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

public class test1 {
    @Test
    void notebookConj(){
        String conj = "(let (mu (sample (normal 0 1))) (observe (normal mu 1) 2.3) mu)";
        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(conj, new Random(0), 60000, 3000);

        float mean = 0f;
        for(Float v : chain) mean += v;
        mean /= chain.size();

        float variance = 0f;
        for(Float v : chain) variance += (v - mean) * (v - mean);
        float std = (float) Math.sqrt(variance / chain.size());

        assertEquals(1.150f, mean, 0.03f);
        assertEquals(0.707f, std, 0.05f);
    }

    @Test
    void notebookBits(){
        StringBuilder letBindings = new StringBuilder();
        for(int i = 1; i <= 8; i++){
            letBindings.append(String.format("b%d (if (sample (bernoulli 0.5)) 1 0) ", i));
        }
        StringBuilder sumTerms = new StringBuilder();
        for(int i = 1; i <= 8; i++){
            sumTerms.append("b").append(i).append(" ");
        }
        String bits = "(let (" + letBindings + "total (+ " + sumTerms + ")) (observe (normal 7 2) total) total)";

        var chain = ProbabilisticProgram.runSSMHProbabilisticProgram(bits, new Random(1), 40000, 3000);

        // exact posterior mean, computed the same way the notebook does:
        // weight k by (8 choose k) * exp(-0.5*((k-7)/2)^2), then take the weighted average
        double numerator = 0, denominator = 0;
        for(int k = 0; k <= 8; k++){
            double weight = binomialCoefficient(8, k) * Math.exp(-0.5 * Math.pow((k - 7) / 2.0, 2));
            numerator += k * weight;
            denominator += weight;
        }
        double exact = numerator / denominator;

        float mean = 0f;
        for(Float v : chain) mean += v;
        mean /= chain.size();

        assertEquals(exact, mean, 0.15);
    }

    private static long binomialCoefficient(int n, int k){
        long result = 1;
        for(int i = 0; i < k; i++){
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
}
