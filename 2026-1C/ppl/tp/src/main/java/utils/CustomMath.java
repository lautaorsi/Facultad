package utils;

import java.util.ArrayList;
import java.util.Random;

public class CustomMath {
    public static float[] softmax(ArrayList<Float> logWeights){
        float max = Float.NEGATIVE_INFINITY;
        for(float lw : logWeights) max = Math.max(max, lw);

        float[] exps = new float[logWeights.size()];
        float sum = 0f;
        for(int i = 0; i < logWeights.size(); i++){
            exps[i] = (float) Math.exp(logWeights.get(i) - max);
            sum += exps[i];
        }
        for(int i = 0; i < exps.length; i++){
            exps[i] /= sum;
        }
        return exps;
    }

    public static int[] categoricalResample(float[] weights, int n, Random rng){
        int[] result = new int[n];
        for(int j = 0; j < n; j++){
            float u = rng.nextFloat();
            float cumulative = 0f;
            int chosen = weights.length - 1;
            for(int i = 0; i < weights.length; i++){
                cumulative += weights[i];
                if(u < cumulative){
                    chosen = i;
                    break;
                }
            }
            result[j] = chosen;
        }
        return result;
    }
}
