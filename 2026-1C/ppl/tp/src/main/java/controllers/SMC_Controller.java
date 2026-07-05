package controllers;
import java.util.ArrayList;
import java.util.Random;

import distributions.Distribution;
import machine.Machine;
import messages.DoneMessage;
import messages.Message;
import messages.ObserveMessage;
import messages.SampleMessage;
import utils.CustomMath;



public class SMC_Controller extends Controller{
        
        ArrayList<Machine> particles;
        int particle_qtty;
        ArrayList<Random> rngs;

        public SMC_Controller(ArrayList<Machine> particles, ArrayList<Random> rngs){//Controller instantiator
            this.particles = particles;
            this.particle_qtty = particles.size();
            this.rngs = rngs;
        }

        
        public ArrayList<Float> runInference(){
            while(true){

                ArrayList<Message> messages = new ArrayList<>();
                
                for(Machine particle : particles){
                    messages.add(this.advance(particle));
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
                        
                        Float log_prob = this.calculateLogDensity(d, observed_value);
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


        private Message advance(Machine machine){
            Message message = machine.resume();

            while(message.isSample()){
                if(message instanceof SampleMessage sampleMessage){
                    Distribution distribution = sampleMessage.distribution();
                    Random rng = machine.getRNG();

                    Float sample = this.sampleFrom(distribution, rng);

                    machine.pushValue(sample);
                } 
                message = machine.resume();
            }
            return message;
            
        }


    }
    
