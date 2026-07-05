package probabilistic_program;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import controllers.Controller;
import controllers.LW_Controller;
import controllers.SMC_Controller;
import controllers.SSMH_Controller;
import distributions.Distribution;
import machine.Machine;
import messages.DoneMessage;
import messages.Message;
import messages.ObserveMessage;
import messages.SampleMessage;
import utils.CustomMath;
import utils.Tuple;


public class ProbabilisticProgram {
    


    public static Tuple<Object, Float> runLWProbabilisticProgram(String program, Random rng){

        Controller lwController = new LW_Controller();

        Machine machine = Machine.initial_machine(program, rng);

        while(true){
            Message message = machine.resume();

            if(message instanceof DoneMessage doneMessage){
                Tuple<Object, Float> return_value =  new Tuple<Object, Float>(doneMessage.value(), machine.getLogW());
                return return_value;
            }
            if(message instanceof SampleMessage sampleMessage){

                Distribution d = sampleMessage.distribution();

                float sampleValue = lwController.sampleFrom(d, rng);

                machine.pushValue(sampleValue);
            }
            if(message instanceof ObserveMessage observeMessage){
                
                Distribution d = observeMessage.distribution();
                Float observed_value = observeMessage.observed_value();

                Float logWIncrement = lwController.calculateLogDensity(d, observed_value);
                
                machine.increaseLogW(logWIncrement);

                machine.pushValue(observed_value);
            }
        }
    }


    public static ArrayList<Float> runSMCProbabilisticProgram(String program, ArrayList<Random> rngs, int particle_qtty){
        ArrayList<Machine> particles = new ArrayList();
        for(Random rng : rngs){//times particles_qtty
            
            //instantiate particle (machine) w/ specific seed and main environment
            Machine particle = Machine.initial_machine(program, rng);

            //add it to particle list
            particles.add(particle);
        }

        Controller smcController = new SMC_Controller(particles);  //We create specific controller (SSMH)

        

        while(true){

            ArrayList<Message> messages = new ArrayList<>();
            
            for(Machine particle : particles){
                messages.add(advance(particle, smcController));
            }    
            

            int doneCount = 0;
            int observeCount = 0;
            for(Message message : messages){
                if(message.isDone()){
                    doneCount += 1;
                }
                if(message.isObserve()){
                    observeCount += 1;
                }
            }

        
            if(doneCount == messages.size()){
                ArrayList<Float> returnList = new ArrayList<>();
                for(Message message : messages){
                    if(message instanceof DoneMessage doneMessage){
                        returnList.add(((Number) doneMessage.value()).floatValue());
                    }
                }
                return returnList;
            }
            if(observeCount != messages.size()){
                throw new Error("Particles reached different breakpoints");
            }


            ArrayList<Float> log_inc = new ArrayList<>();
            ArrayList<Machine> paused = new ArrayList<>();
        
            for(Message message : messages){
                
                Distribution d;
                Float observed_value;
                Machine machine;
                if(message instanceof ObserveMessage observeMessage){
                    d = observeMessage.distribution();
                    observed_value = observeMessage.observed_value();
                    machine = observeMessage.machine();
                    
                    Float log_prob = smcController.calculateLogDensity(d, observed_value);
                    machine.increaseLogW(log_prob);
                    
                    log_inc.add(log_prob);
                    machine.pushValue(observed_value);
                    paused.add(machine);
                }
            }
        
            int[] anc = CustomMath.categoricalResample(CustomMath.softmax(log_inc), particle_qtty, rngs.get(0));

            ArrayList<Machine> nextParticles = new ArrayList<>();
            for(int j = 0; j < particle_qtty; j++){
                Machine ancestor = paused.get(anc[j]);
                nextParticles.add(ancestor.fork(rngs.get(j)));
            }
            particles = nextParticles;
        }


    }




    public static ArrayList<Float> runSSMHProbabilisticProgram(String program, Random rng, int steps, int warmup){

        Controller ssmhController = new SSMH_Controller();

        HashMap<Object, Object> cache = new HashMap<>();

        ArrayList<Object> valueXSO = runSingleSSMHExecution(program, rng, null, cache, ssmhController); 

        Object value = valueXSO.get(0);

        HashMap<Object,Object> X = (HashMap<Object,Object>) valueXSO.get(1);

        HashMap<Object,Float> S = (HashMap<Object,Float>) valueXSO.get(2);

        HashMap<Object,Float> O = (HashMap<Object,Float>) valueXSO.get(3);

        ArrayList<Float> chain = new ArrayList<Float>();

        ArrayList<Object> keys = new ArrayList<>(X.keySet());

        for(int i = 0; i < steps+warmup; i++){
            Object a0 =  keys.get(rng.nextInt(keys.size()));
        
            ArrayList value2X2S2O2 = runSingleSSMHExecution(program, rng, a0, X, ssmhController);
            
            Object value2 = value2X2S2O2.get(0);

            HashMap<Object,Object> X2 = (HashMap<Object,Object>) value2X2S2O2.get(1);

            HashMap<Object,Float> S2 = (HashMap<Object,Float>) value2X2S2O2.get(2);

            HashMap<Object,Float> O2 = (HashMap<Object,Float>) value2X2S2O2.get(3);
        
            if(Math.log(rng.nextDouble()) < mh_log_alpha(X,X2,S,S2,O,O2,a0)){
                value = value2;
                X = X2;
                S = S2;
                O = O2; 
            }
            if( i >= warmup){
                chain.add(((Number) value).floatValue());; 
            }
        }
        
        return chain;
    }






    private static ArrayList runSingleSSMHExecution(String program, Random rng, Object x0, HashMap<Object, Object> cache, Controller controller){
        Machine machine = Machine.initial_machine(program, rng);
        
        HashMap<Object,Object> X = new HashMap<>();
        HashMap<Object,Float> S = new HashMap<>();
        HashMap<Object,Float> O = new HashMap<>();

        
        while(true){
            Message message = machine.resume();

            if(message instanceof SampleMessage sampleMessage){
                
                Object address = sampleMessage.address();
                
                Distribution distribution = sampleMessage.distribution();

               Float x = (address.equals(x0) || !cache.containsKey(address)) ? distribution.sample(rng) : ((Number) cache.get(address)).floatValue();
            
                X.put(address, x);



                S.put(address, controller.calculateLogDensity(distribution, x));
                
                machine.pushValue(x);
            }
            if(message instanceof ObserveMessage observeMessage){

                Object address = observeMessage.address();

                Distribution distribution = observeMessage.distribution();

                Float observedValue = observeMessage.observed_value();

                O.put(address, controller.calculateLogDensity(distribution, observedValue));

                machine.pushValue(observedValue);
            }
            if(message instanceof DoneMessage doneMessage){
                
                ArrayList returnList = new ArrayList<>();

                returnList.add(doneMessage.value());
                returnList.add(X);
                returnList.add(S);
                returnList.add(O);

                return returnList;
            }
        }
    }



    private static double mh_log_alpha(HashMap<Object,Object> X,HashMap<Object,Object> X2, HashMap<Object,Float> S, HashMap<Object,Float> S2, HashMap<Object,Float> O, HashMap<Object,Float> O2, Object a0){
        Set<Object> fwd = new HashSet<>(X2.keySet());
        fwd.removeAll(X.keySet());
        fwd.add(a0);

        Set<Object> rev = new HashSet<>(X.keySet());
        rev.removeAll(X2.keySet());
        rev.add(a0);

        double num = 0.0;
        for(Map.Entry<Object, Float> entry : S2.entrySet()){
            if(!fwd.contains(entry.getKey())){
                num += entry.getValue();
            }
        }
        for(float o2 : O2.values()){
            num += o2;
        }

        double den = 0.0;
        for(Map.Entry<Object, Float> entry : S.entrySet()){
            if(!rev.contains(entry.getKey())){
                den += entry.getValue();
            }
        }
        for(float o : O.values()){
            den += o;
        }

        return (Math.log(X.size()) - Math.log(X2.size())) + (num - den);
    }
















     

    private static Message advance(Machine machine, Controller controller){
        Message message = machine.resume();

        while(message.isSample()){
            if(message instanceof SampleMessage sampleMessage){
                Distribution distribution = sampleMessage.distribution();
                Random rng = machine.getRNG();

                Float sample = controller.sampleFrom(distribution, rng);

                machine.pushValue(sample);
            } 
            message = machine.resume();
        }
        return message;
        
    }
}
