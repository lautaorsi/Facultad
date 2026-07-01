import java.util.ArrayList;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.List;

import javax.crypto.EncryptedPrivateKeyInfo;

import java.beans.Expression;
import java.util.ArrayDeque;

public class Machine{

    private ArrayDeque<Continuation> controlStack;
    private ArrayDeque valueStack; //Originalmente iba a usar Stack, pero aparentemente Java recomienda usar ArrayDeque
    private HashMap env;
    private int seed;
    private float log_w;

    //Q? Creo que no hace falta poner los op. ternarios ya que la única llamada a Machine instantiator 
    // ocurre desde initial_machine y es controlada (i.e. nunca va a pasar new_env = null)
    private Machine(ArrayDeque<Continuation> controlStack, ArrayDeque valueStack, HashMap env, Int seed, Float log_w){
        self.controlStack = controlStack;
        
        self.valueStack = (valueStack != null) ? valueStack : new ArrayDeque<>();
        
        self.env = (env != null) ? env : new HashMap<>();
        
        self.seed = seed; 

        self.log_w = (log_w != null) ? log_w : 0.0f;
    }


    public Machine instantiate_initial_machine(Int seed, HashMap env, Expression main){
        
        ArrayDeque<Continuation> controlStack = new ArrayDeque<Continuation>();
        List emptyAddress = new List();
        Ev newEv = new Ev(main, env, emptyAddress);
        controlStack.push(newEv);

        return Machine(controlStack,null, env, seed, null);
    }

    public Machine fork(Machine self,Int seed){ // Forking is copying the stacks
        return Machine(controlStack=self.controlStack,
                        valueStack=self.valueStack,
                        env=self.env,
                        seed=self.seed,
                        log_w=self.log_w);
    }

    public void executeNextInstruction(){
        Continuation continuation = C.pop();

        continuation.executeOn(self);
    }

    public boolean hasInstructions(){
        return !controlStack.isEmpty();
    }

    public Continuation getNextControl(){
        return controlStack.pop();
    }

    public pushInstruction(Continuation continuation){
        controlStack.push(continuation);
    }

    public getNextValue(){
        return valueStack.pop();
    }

    public void pushValue(T value){
        valueStack.push(value);
    }

    public void push_body(List body, HashMap environment, List address){
        seq = new List();

        Ev ev;
        Discard discard = new Discard();
        for(int i = 0; i < body.length() -1; i++){
            List newAddress = new List(address);
            newAdress.add("body", n);
            ev = new Ev(body.get(i), environment, newAdress);
            seq.add(ev);
            seq.add(discard);
        }

        List newAddress = new List(address);
        newAddress.add("body", body.length() -1); 
        ev = new Ev(body.get(-1), environment, newAddress);
    
        for(int i = seq.length() -1; i >= 0; i--){
            controlStack.push(seq.get(i));
        }
    }







}