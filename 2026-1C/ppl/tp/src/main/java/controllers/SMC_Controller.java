package controllers;
import java.util.ArrayList;

import machine.Machine;



public class SMC_Controller extends Controller{
        
        ArrayList<Machine> particles;
        int particles_qtty;

        public SMC_Controller(ArrayList<Machine> particles){//Controller instantiator
            this.particles = particles;
            this.particles_qtty = particles.size();
        }
        
}
    
