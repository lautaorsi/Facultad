package probabilistic_program;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Random;

import org.junit.jupiter.api.Test;

public class test2 {
    @Test
    void threeControllersAgree(){
        String conj = "(let (mu (sample (normal 0 1))) (observe (normal mu 1) 2.3) mu)";

        // LW
        ArrayList<Float> lwValues = new ArrayList<>();
        ArrayList<Float> lwLogWeights = new ArrayList<>();
        for(int i = 0; i < 100000; i++){
            var result = ProbabilisticProgram.runLWProbabilisticProgram(conj, new Random(2000 + i));
            lwValues.add(((Number) result.first()).floatValue());
            lwLogWeights.add(result.second());
        }
        float lwMean = weightedMean(lwValues, lwLogWeights);

        // SMC
        ArrayList<Random> smcRngs = new ArrayList<>();
        for(int i = 0; i < 20000; i++) smcRngs.add(new Random(1000 + i));
        var smcResults = ProbabilisticProgram.runSMCProbabilisticProgram(conj, smcRngs, 20000);
        float smcMean = 0f;
        for(Float v : smcResults) smcMean += v;
        smcMean /= smcResults.size();

        // SSMH
        var ssmhChain = ProbabilisticProgram.runSSMHProbabilisticProgram(conj, new Random(0), 60000, 3000);
        float ssmhMean = 0f;
        for(Float v : ssmhChain) ssmhMean += v;
        ssmhMean /= ssmhChain.size();

        assertEquals(1.150f, lwMean, 0.03f);
        assertEquals(1.150f, smcMean, 0.2f);
        assertEquals(1.150f, ssmhMean, 0.03f);
    }

    private static float weightedMean(ArrayList<Float> values, ArrayList<Float> logWeights){
        float maxLogW = Float.NEGATIVE_INFINITY;
        for(float lw : logWeights) maxLogW = Math.max(maxLogW, lw);

        float sumWeights = 0f;
        float weightedSum = 0f;
        for(int i = 0; i < values.size(); i++){
            float w = (float) Math.exp(logWeights.get(i) - maxLogW);
            sumWeights += w;
            weightedSum += w * values.get(i);
        }
        return weightedSum / sumWeights;
    }
}
