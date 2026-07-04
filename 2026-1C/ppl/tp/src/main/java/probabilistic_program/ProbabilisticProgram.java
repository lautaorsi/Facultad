package probabilistic_program;

import java.util.ArrayList;
import java.util.Random;

import controllers.Controller;
import controllers.LW_Controller;
import controllers.SMC_Controller;
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
