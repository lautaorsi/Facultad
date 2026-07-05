package distributions;

import java.util.Random;

public class Bernoulli extends Distribution {
    private final float p;

    public Bernoulli(float p){
        this.p = p;
    }

    @Override
    public Float logProb(Float x){
        boolean value = x != 0f;
        return (float) Math.log(value ? p : (1 - p));
    }

    @Override
    public float sample(Random rng){
        return rng.nextDouble() < p ? 1.0f : 0.0f;
    }
}